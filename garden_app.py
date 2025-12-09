import os
from datetime import datetime
import io
import tempfile

import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import boto3
from dotenv import load_dotenv, dotenv_values
from PIL import Image
import numpy as np

# ----------------- ENV LOADING -----------------

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH)

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

st.set_page_config(layout="wide")

# ----------------- DB HELPERS -----------------


def get_connection():
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        raise RuntimeError("Database environment variables are not fully set.")

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )
    return conn


def fetch_trays():
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT tray_id, name, rows, cols, started_at
                    FROM trays
                    ORDER BY name;
                    """
                )
                return cur.fetchall()
    finally:
        conn.close()


def fetch_seeds():
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT seed_id, variety_name, species, vendor, notes, cast(year as int) as year
                    FROM seeds
                    ORDER BY species, variety_name;
                    """
                )
                return cur.fetchall()
    finally:
        conn.close()


def insert_seed(variety_name, species, vendor, notes, year):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO seeds (variety_name, species, vendor, notes, year)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING seed_id;
                    """,
                    (variety_name, species, vendor, notes, year),
                )
                row = cur.fetchone()   # RealDictCursor → dict
                return row["seed_id"]
    finally:
        conn.close()

def fetch_vendors():
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT vendor FROM seeds WHERE vendor IS NOT NULL ORDER BY vendor;"
                )
                rows = cur.fetchall()
                return [r["vendor"] for r in rows]
    finally:
        conn.close()


def fetch_existing_seedlings(tray_id):
    """
    Returns dict: (row_index, col_index) -> seed_id
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT row_index, col_index, seed_id
                    FROM seedlings
                    WHERE tray_id = %s;
                    """,
                    (tray_id,),
                )
                rows = cur.fetchall()
    finally:
        conn.close()

    mapping = {}
    for row in rows:
        mapping[(row["row_index"], row["col_index"])] = row["seed_id"]
    return mapping


def upsert_seedlings(tray_id, rows, cols, grid_selection):
    """
    grid_selection[(r, c)] = seed_id or None
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for r in range(rows):
                    for c in range(cols):
                        seed_id = grid_selection.get((r, c))
                        if seed_id is None:
                            cur.execute(
                                """
                                DELETE FROM seedlings
                                WHERE tray_id = %s AND row_index = %s AND col_index = %s;
                                """,
                                (tray_id, r, c),
                            )
                        else:
                            cur.execute(
                                """
                                INSERT INTO seedlings (tray_id, row_index, col_index, seed_id)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (tray_id, row_index, col_index)
                                DO UPDATE SET seed_id = EXCLUDED.seed_id;
                                """,
                                (tray_id, r, c, seed_id),
                            )
    finally:
        conn.close()


def insert_tray_image(tray_id, s3_url, original_filename, taken_at):
    if taken_at is None:
        taken_at = datetime.utcnow()

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tray_images (tray_id, s3_url, taken_at, original_filename)
                    VALUES (%s, %s, %s, %s)
                    RETURNING tray_image_id;
                    """,
                    (tray_id, s3_url, taken_at, original_filename),
                )
                row = cur.fetchone()
                return row["tray_image_id"]
    finally:
        conn.close()


def fetch_seedling_map(tray_id):
    """
    Returns dict: (row_index, col_index) -> seedling_id
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT seedling_id, row_index, col_index
                    FROM seedlings
                    WHERE tray_id = %s;
                    """,
                    (tray_id,),
                )
                rows = cur.fetchall()
    finally:
        conn.close()

    mapping = {}
    for row in rows:
        mapping[(row["row_index"], row["col_index"])] = row["seedling_id"]
    return mapping


def upsert_seedling_measurements(
    tray_id,
    tray_image_id,
    green_matrix,
    center_matrix,
    germ_matrix,
):
    seedling_map = fetch_seedling_map(tray_id)
    rows = len(green_matrix)
    cols = len(green_matrix[0]) if rows > 0 else 0

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for r in range(rows):
                    for c in range(cols):
                        seedling_id = seedling_map.get((r, c))
                        if seedling_id is None:
                            continue

                        overall = float(green_matrix[r][c])
                        center = float(center_matrix[r][c])
                        germ = bool(germ_matrix[r][c])

                        cur.execute(
                            """
                            INSERT INTO seedling_measurements
                                (seedling_id, tray_image_id,
                                 overall_green_ratio, center_green_ratio, germinated)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (seedling_id, tray_image_id) DO UPDATE
                                SET overall_green_ratio = EXCLUDED.overall_green_ratio,
                                    center_green_ratio  = EXCLUDED.center_green_ratio,
                                    germinated          = EXCLUDED.germinated;
                            """,
                            (seedling_id, tray_image_id, overall, center, germ),
                        )
    finally:
        conn.close()


# ----------------- S3 HELPERS -----------------


def get_s3_client():
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_REGION]):
        raise RuntimeError(
            "AWS credentials or region not set. "
            "Ensure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_REGION/AWS_DEFAULT_REGION are in .env."
        )

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
    )
    return s3


def upload_file_to_s3(file_bytes, filename, content_type, tray_id):
    """
    file_bytes: raw image bytes
    filename: original filename
    content_type: MIME type
    tray_id: used in key
    """
    s3 = get_s3_client()

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = filename.replace(" ", "_")
    key = f"trays/{tray_id}/{date_str}/{timestamp}_{safe_name}"

    s3.upload_fileobj(
        Fileobj=io.BytesIO(file_bytes),
        Bucket=S3_BUCKET_NAME,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )

    s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{key}"
    return s3_url, safe_name


# ----------------- ANALYSIS HELPERS (PIL + NumPy) -----------------

GERM_NOISE_FLOOR = 0.01
WEAK_THRESHOLD = 0.01
STRONG_THRESHOLD = 0.09
CENTER_FRAC = 0.25




def split_tray_equal_grid(image_path, rows, cols, inner_margin_frac=0.15):
    """
    Load image, split into equal grid. Returns [((r, c), PIL.Image cell), ...]
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    cell_h = h // rows
    cell_w = w // cols

    cells = []
    for r in range(rows):
        for c in range(cols):
            y1_outer = r * cell_h
            y2_outer = (r + 1) * cell_h
            x1_outer = c * cell_w
            x2_outer = (c + 1) * cell_w

            margin_y = int((y2_outer - y1_outer) * inner_margin_frac)
            margin_x = int((x2_outer - x1_outer) * inner_margin_frac)

            y1 = y1_outer + margin_y
            y2 = y2_outer - margin_y
            x1 = x1_outer + margin_x
            x2 = x2_outer - margin_x

            cell = img.crop((x1, y1, x2, y2))
            cells.append(((r, c), cell))

    return cells


def green_overall_and_center(cell_img, noise_floor=GERM_NOISE_FLOOR, center_frac=CENTER_FRAC):
    """
    cell_img: PIL Image (RGB). Returns (overall_ratio, center_ratio).
    """
    hsv = cell_img.convert("HSV")
    hsv_np = np.array(hsv)

    H = hsv_np[:, :, 0].astype(np.float32)
    S = hsv_np[:, :, 1].astype(np.float32)
    V = hsv_np[:, :, 2].astype(np.float32)

    # Approx mapping of OpenCV [35,85] to Pillow H [0,255]
    lower_h = 50
    upper_h = 120
    lower_s = 60
    lower_v = 80

    green_mask = (
        (H >= lower_h) & (H <= upper_h) &
        (S >= lower_s) &
        (V >= lower_v)
    )

    total_pixels = green_mask.size
    green_pixels = np.count_nonzero(green_mask)

    if total_pixels == 0:
        overall_ratio = 0.0
    else:
        overall_ratio = green_pixels / float(total_pixels)
        if overall_ratio < noise_floor:
            overall_ratio = 0.0

    h, w = H.shape
    ch = int(h * center_frac)
    cw = int(w * center_frac)
    y1 = (h - ch) // 2
    y2 = y1 + ch
    x1 = (w - cw) // 2
    x2 = x1 + cw

    center_mask = green_mask[y1:y2, x1:x2]
    center_total = center_mask.size
    center_green = np.count_nonzero(center_mask)

    if center_total == 0:
        center_ratio = 0.0
    else:
        center_ratio = center_green / float(center_total)
        if center_ratio < noise_floor:
            center_ratio = 0.0

    return overall_ratio, center_ratio


def compute_germination_simple(cells, rows, cols,
                               weak_threshold=WEAK_THRESHOLD,
                               strong_threshold=STRONG_THRESHOLD,
                               noise_floor=GERM_NOISE_FLOOR,
                               center_frac=CENTER_FRAC):
    green_matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
    center_matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
    germ_matrix = [[False for _ in range(cols)] for _ in range(rows)]

    for (r, c), cell_img in cells:
        overall, center = green_overall_and_center(
            cell_img,
            noise_floor=noise_floor,
            center_frac=center_frac,
        )
        green_matrix[r][c] = overall
        center_matrix[r][c] = center

        germ_matrix[r][c] = (
            (overall >= strong_threshold) or
            (overall >= weak_threshold and center > 0.0)
        )

    return green_matrix, center_matrix, germ_matrix


def analyze_image_and_save(tray_id,
                           tray_image_id,
                           file_bytes,
                           tray_rows,
                           tray_cols):
    """
    Saves bytes to temp file, runs analysis, writes seedling_measurements.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        cells = split_tray_equal_grid(tmp_path, tray_rows, tray_cols)
        green_matrix, center_matrix, germ_matrix = compute_germination_simple(
            cells,
            rows=tray_rows,
            cols=tray_cols,
        )
        upsert_seedling_measurements(
            tray_id=tray_id,
            tray_image_id=tray_image_id,
            green_matrix=green_matrix,
            center_matrix=center_matrix,
            germ_matrix=germ_matrix,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return green_matrix, center_matrix, germ_matrix


# ----------------- STREAMLIT APP -----------------

st.title("🌿 Seedling Tracker")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
    st.error("Database environment variables are not fully set. Check your .env.")
    st.stop()

tabs = st.tabs(["Tray → Seeds", "Upload & Analyze", "Manage Seeds", "Manage Trays"])

# ---------- TAB 1: TRAY → SEEDS ----------

with tabs[0]:
    st.header("Tray → Seed Mapping")

    trays = fetch_trays() 
    if not trays:
        st.error("No trays found in `trays` table.")
    else:
        tray_options = {f"{t['tray_id']} – {t['name']} ({t['rows']}x{t['cols']})": t for t in trays}
        selected_tray_label = st.selectbox("Select a tray", list(tray_options.keys()))
        selected_tray = tray_options[selected_tray_label]

        tray_id = selected_tray["tray_id"]
        tray_rows = selected_tray["rows"]
        tray_cols = selected_tray["cols"]
 
        #st.write(
        #    f"**Tray:** {selected_tray['name']} (ID {tray_id}) – "
        #    f"{tray_rows} rows × {tray_cols} cols"
        #)

        seeds = fetch_seeds()
        if not seeds:
            st.warning("No seeds found in `seeds` table. Use the 'Manage Seeds' tab to add some.")
        else:
            seed_display_to_id = {"(Empty)": None}
            for s in seeds:
                parts = [s["species"]]
                if s.get("variety_name"):
                    parts.append(s["variety_name"])
                if s.get("vendor"):
                    parts.append(s["vendor"])
                label = f" | ".join(parts)
                seed_display_to_id[label] = s["seed_id"]
            seed_id_to_display = {v: k for k, v in seed_display_to_id.items() if v is not None}

            existing = fetch_existing_seedlings(tray_id)
            st.subheader("Assign or update seeds to each cell")
            

            grid_selection = {}
            for r in range(tray_rows):
                cols_streamlit = st.columns(tray_cols)
                for c in range(tray_cols):
                    with cols_streamlit[c]:
                        key = f"cell_{tray_id}_{r}_{c}"
                        label = f"({r}, {c})"

                        default_seed_id = existing.get((r, c))
                        if default_seed_id is None:
                            default_label = "(Empty)"
                        else:
                            default_label = seed_id_to_display.get(default_seed_id, "(Empty)")

                        choice = st.selectbox(
                            label,
                            options=list(seed_display_to_id.keys()),
                            index=list(seed_display_to_id.keys()).index(default_label),
                            key=key,
                        )

                        seed_id = seed_display_to_id[choice]
                        grid_selection[(r, c)] = seed_id

            st.markdown("---")
            if st.button("💾 Save mapping to `seedlings` table"):
                upsert_seedlings(tray_id, tray_rows, tray_cols, grid_selection)
                st.success("Seedlings table updated for this tray.")

# ---------- TAB 2: UPLOAD & ANALYZE ----------

with tabs[1]:
    st.header("Upload Tray Image & Analyze")

    if not S3_BUCKET_NAME:
        st.error("S3_BUCKET_NAME is not set in .env.")
    elif not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        st.error("AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not set in .env.")
    else:
        trays2 = fetch_trays()
        if not trays2:
            st.error("No trays found in `trays` table.")
        else:
            tray_options2 = {f"{t['tray_id']} – {t['name']} ({t['rows']}x{t['cols']})": t for t in trays2}
            selected_tray_label2 = st.selectbox("Select tray for this image", list(tray_options2.keys()))
            selected_tray2 = tray_options2[selected_tray_label2]
            tray_id2 = selected_tray2["tray_id"]
            tray_rows2 = selected_tray2["rows"]
            tray_cols2 = selected_tray2["cols"]

            st.write(
                f"Uploading image for **{selected_tray2['name']}** "
                f"(ID {tray_id2}, {tray_rows2}×{tray_cols2})."
            )

            uploaded_file = st.file_uploader(
                "Upload a tray image (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
            )

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Preview", use_column_width=True)

                st.caption("Optionally specify when the photo was taken (defaults to now, UTC).")
                taken_at_override = None

                if st.button("Upload to S3, save, and analyze"):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        filename = uploaded_file.name
                        content_type = uploaded_file.type

                        with st.spinner("Uploading to S3..."):
                            s3_url, safe_name = upload_file_to_s3(
                                file_bytes=file_bytes,
                                filename=filename,
                                content_type=content_type,
                                tray_id=tray_id2,
                            )

                        with st.spinner("Saving tray_images record..."):
                            tray_image_id = insert_tray_image(
                                tray_id=tray_id2,
                                s3_url=s3_url,
                                original_filename=safe_name,
                                taken_at=None,
                            )

                        with st.spinner("Running germination analysis..."):
                            green_matrix, center_matrix, germ_matrix = analyze_image_and_save(
                                tray_id=tray_id2,
                                tray_image_id=tray_image_id,
                                file_bytes=file_bytes,
                                tray_rows=tray_rows2,
                                tray_cols=tray_cols2,
                            )

                        st.success("✅ Image uploaded, recorded, and analyzed!")
                        st.write(f"**tray_image_id:** {tray_image_id}")
                        st.write(f"**S3 URL:** {s3_url}")

                        total_cells = tray_rows2 * tray_cols2
                        germinated_cells = sum(
                            1
                            for r in range(tray_rows2)
                            for c in range(tray_cols2)
                            if germ_matrix[r][c]
                        )
                        st.write(f"Germinated cells: {germinated_cells} / {total_cells}")

                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
            else:
                st.info("Upload an image to get started.")

# ---------- TAB 3: MANAGE SEEDS ----------

# ---------- TAB: MANAGE SEEDS ----------

with tabs[2]:
    st.header("Manage Seeds")

    # ---- reset logic for the add-seed form ----
    if "reset_seed_form" not in st.session_state:
        st.session_state.reset_seed_form = False

    if st.session_state.reset_seed_form:
        for key in ["variety_name_input", "species_input", "vendor_input", "year", "notes_input"]:
            if key in st.session_state:
                # empty string for text, 0-ish stays as is for numbers
                if key in ["variety_name_input", "species_input", "vendor_input", "notes_input"]:
                    st.session_state[key] = ""
        st.session_state.reset_seed_form = False

    st.subheader("Add a new seed")

    with st.form("add_seed_form"):
        variety_name = st.text_input("Variety Name *", key="variety_name_input")
        species = st.text_input("Species", key="species_input")
        # Autofill vendor dropdown
        existing_vendors = fetch_vendors()
        vendor_options = existing_vendors + ["Add new vendor..."]

        vendor_choice = st.selectbox(
            "Vendor",
            vendor_options,
            key="vendor_select_input"
        )

        if vendor_choice == "Add new vendor...":
            vendor = st.text_input("Enter new vendor", key="vendor_input")
        else:
            vendor = vendor_choice  # selected existing vendor
        
        year_purchased = st.number_input(
            "Year Purchased",
            min_value=2023,
            max_value=2026,
            step=1,
            key="year",
        )
        notes = st.text_area("Notes", key="notes_input")

        submitted = st.form_submit_button("Add seed")

        if submitted:
            if not st.session_state.variety_name_input.strip():
                st.error("Variety name is required.")
            else:
                try:
                    seed_id = insert_seed(
                        variety_name=st.session_state.variety_name_input.strip(),
                        species=st.session_state.species_input.strip() or None,
                        vendor=vendor or None,
                        notes=st.session_state.notes_input.strip() or None,
                        year=int(st.session_state.year),
                    )
                    st.success(f"Seed added with ID {seed_id}.")

                    # clear on next rerun
                    st.session_state.reset_seed_form = True
                    st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error inserting seed: {e}")

    st.subheader("Existing seeds")

    seeds3 = fetch_seeds()
    if not seeds3:
        st.info("No seeds in the table yet.")
    else:
        display_rows = []
        for s in seeds3:
            year_val = s.get("year")
            if year_val is None:
                year_display = ""
            else:
                year_display = str(int(year_val))  # force clean display

            display_rows.append(
                {
                   "Species": s.get("species"),
                    "Variety": s["variety_name"],
                    "Vendor": s.get("vendor"),
                    "Year Purchased": year_display,
                    
                }
            )
        st.dataframe(display_rows, hide_index= True, width = 800, height = 2900)

# ---------- NEW TAB: MANAGE TRAYS ----------

with tabs[3]:
    st.header("Manage Trays")

    # ---- reset logic for the add-tray form ----
    if "reset_tray_form" not in st.session_state:
        st.session_state.reset_tray_form = False

    if st.session_state.reset_tray_form:
        # Clear text inputs; number inputs reset automatically
        for key in ["tray_name_input", "tray_notes_input"]:
            if key in st.session_state:
                st.session_state[key] = ""
        st.session_state.reset_tray_form = False


    # ---- helper to insert a tray ----
    def insert_tray(name, rows, cols, started_at, notes):
        conn = get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO trays (name, rows, cols, started_at, notes)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING tray_id;
                        """,
                        (name, rows, cols, started_at, notes),
                    )
                    return cur.fetchone()["tray_id"]
        finally:
            conn.close()


    st.subheader("Add a new tray")

    with st.form("add_tray_form"):
        tray_name = st.text_input("Tray Name *", key="tray_name_input")

        tray_rows = st.number_input(
            "Rows *",
            min_value=1,
            max_value=100,
            step=1,
            key="tray_rows_input",
        )
        tray_cols = st.number_input(
            "Columns *",
            min_value=1,
            max_value=100,
            step=1,
            key="tray_cols_input",
        )

        tray_started_at = st.date_input(
            "Started At (optional)",
            value=None,
            key="tray_started_at_input"
        )

        tray_notes = st.text_area("Notes", key="tray_notes_input")

        submitted_tray = st.form_submit_button("Add tray")

        if submitted_tray:
            if not st.session_state.tray_name_input.strip():
                st.error("Tray name is required.")
            else:
                try:
                    tray_id = insert_tray(
                        name=st.session_state.tray_name_input.strip(),
                        rows=int(st.session_state.tray_rows_input),
                        cols=int(st.session_state.tray_cols_input),
                        started_at=tray_started_at if tray_started_at else None,
                        notes=st.session_state.tray_notes_input.strip() or None,
                    )

                    st.success(f"Tray added with ID {tray_id}.")

                    # 🔥 CLEAR FORM INPUTS ON NEXT RERUN
                    st.session_state.reset_tray_form = True
                    st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error inserting tray: {e}")


    st.subheader("Existing trays")
    trays_list = fetch_trays()

    if not trays_list:
        st.info("No trays in the table yet.")
    else:
        st.dataframe(
            [
                {
                   
                    "Name": t["name"],
                    "Rows": t["rows"],
                    "Cols": t["cols"],
                    "Started At": t.get("started_at"),
                    
                }
                for t in trays_list
            ]
        , width = 800, height = 200, hide_index=True)