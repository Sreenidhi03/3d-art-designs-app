import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import tempfile
import cv2
import trimesh
import plotly.graph_objects as go
from stl import mesh as stl_mesh

st.set_page_config(layout="wide", page_title="Art")

st.title("⚡ Art Design")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Image Generation",
    "⚙️ STL Generator",
    "🖍️ G-Code",
    "📐 Stl Image",
    "📂 Report"
])

# ------------------ TAB 1 - Upload & Relief Generator ------------------
with tab1:
    st.header("🖼️ Height map")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.session_state['original_image'] = image
        st.image(image, caption="Original Image", use_column_width=True)

        image_array = np.array(image)
        if len(image_array.shape) == 3:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_array

        gray_image = gray_image.astype(float)

        colormap_options = ["gray", "viridis", "plasma", "inferno", "magma", "cividis"]
        selected_colormap = st.selectbox("🎨 Select Colormap for Height Map Preview", colormap_options, index=0)

        fig, ax = plt.subplots()
        im = ax.imshow(gray_image, cmap=selected_colormap)
        ax.set_title("Grayscale Height Map Preview")
        st.pyplot(fig)

        st.session_state['relief'] = gray_image


# ------------------ TAB 2 - STL Generator --
with tab2:
    st.header("🧊 STL Generator")

    if 'relief' in st.session_state:
        relief = st.session_state['relief']
        rows, cols = relief.shape
        st.write(f"Relief Shape: {rows} x {cols}")

        base_thickness = st.number_input("Base Thickness (mm)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        add_base_plate = st.checkbox("Add Base Plate", value=True)
        scale_percent = st.slider("Preview Resolution Scale (%)", 10, 100, 50, 10)

        # Downscale for preview
        scaled_relief = cv2.resize(relief, (0, 0), fx=scale_percent/100, fy=scale_percent/100, interpolation=cv2.INTER_AREA)

        # Colormap and Preview Mode Selection
        colormap_options = ["gray", "viridis", "plasma", "inferno", "magma", "cividis"]
        selected_colormap = st.selectbox("🎨 Select Colormap for 3D STL Preview", colormap_options, index=1)
        preview_mode = st.selectbox("🗂️ Preview Mode", ["Surface", "Wireframe", "Heightmap"], index=0)

        def generate_stl_fast(relief_data, base_thickness=2.0, add_base=True):
            relief_data = np.flipud(relief_data)
            rows, cols = relief_data.shape
            x = np.linspace(0, cols, cols)
            y = np.linspace(0, rows, rows)
            X, Y = np.meshgrid(x, y)
            Z = relief_data.astype(float)
            if add_base:
                Z += base_thickness
            vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
            faces = []
            for i in range(rows - 1):
                for j in range(cols - 1):
                    idx = i * cols + j
                    faces.append([idx, idx + 1, idx + cols])
                    faces.append([idx + 1, idx + cols + 1, idx + cols])
            faces = np.array(faces)
            mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh_obj, X, Y, Z

        # Generate preview mesh
        preview_mesh, X, Y, Z = generate_stl_fast(scaled_relief, base_thickness, add_base=add_base_plate)

        # Interactive 3D Plotly Preview
        if preview_mode == "Surface":
            plotly_fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=selected_colormap)])
        elif preview_mode == "Wireframe":
            plotly_fig = go.Figure(data=[go.Surface(
                z=Z, x=X, y=Y, colorscale=selected_colormap, showscale=False,
                contours={"z": {"show": True, "start": np.min(Z), "end": np.max(Z), "size": 1}}
            )])
        elif preview_mode == "Heightmap":
            plotly_fig = go.Figure(data=[go.Contour(z=Z, x=X[0], y=Y[:,0], colorscale=selected_colormap)])

        plotly_fig.update_layout(
            title=f"Interactive 3D Relief Preview ({preview_mode})",
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            ),
            width=900,
            height=700,
            margin=dict(l=10, r=10, b=10, t=30)
        )
        st.plotly_chart(plotly_fig)

        # Export full-resolution STL file
        full_mesh, _, _, _ = generate_stl_fast(relief, base_thickness, add_base=add_base_plate)
        stl_path = tempfile.NamedTemporaryFile(delete=False, suffix=".stl").name
        full_mesh.export(stl_path)

        with open(stl_path, "rb") as f:
            st.download_button("📥 Download High-Res STL", f, file_name="relief_output.stl")

    else:
        st.warning("⚠️ Please generate a relief in Tab 1 first.")



# ------------------ TAB 3 - G-code Generator ------------------
with tab3:
    st.header("📐 G-code Generator")
    if 'relief' in st.session_state:
        tool_diameter = st.selectbox("Tool Diameter (mm)", [2, 4, 6, 8])
        feed_rate = st.number_input("Feed Rate (mm/min)", 100, 5000, 1200, 100)
        step_over = st.number_input("Step Over (mm)", 0.1, float(tool_diameter), 1.0, 0.1)
        max_depth = st.number_input("Max Depth (mm)", 0.1, 10.0, 2.0, 0.1)

        relief = st.session_state['relief']
        gcode = ["G21 ; Set units to mm", "G90 ; Absolute positioning", "G1 Z5.0 F500 ; Lift tool"]

        step_px = max(1, int(step_over))
        for i in range(0, relief.shape[0], step_px):
            row = relief[i]
            for j in range(0, relief.shape[1]):
                x = j
                y = i
                z = -min(row[j], max_depth)
                gcode.append(f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{feed_rate}")

        gcode.append("G1 Z5.0 ; Lift tool at end")
        gcode.append("M30 ; End of program")
        gcode_output = "".join(gcode)
        st.code(gcode_output, language='gcode')
        st.download_button("📥 Download G-code File", gcode_output.encode(), file_name="relief_output.nc")
    else:
        st.warning("⚠️ Generate relief first in Tab 1.")

# ------------------ TAB 5 - Batch Relief Generator ------------------
with tab4:
    st.header("📂 Batch Image to STL Relief Generator")

    uploaded_files = st.file_uploader("📤 Upload Multiple Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        width = st.number_input("📏 Width (mm)", min_value=10, max_value=1000, value=100)
    with col2:
        height = st.number_input("📐 Height (mm)", min_value=10, max_value=1000, value=100)
    with col3:
        depth = st.slider("🔽 Relief Depth (mm)", min_value=1, max_value=100, value=10)

    colorscale_options = {
        "Viridis": "Viridis",
        "Cividis": "Cividis",
        "Inferno": "Inferno",
        "Plasma": "Plasma",
        "Magma": "Magma",
        "Greys": "Greys",
        "YlGnBu": "YlGnBu",
        "YlOrRd": "YlOrRd"
    }
    selected_colorscale = st.selectbox("🖌️ 3D Preview Colorscale", list(colorscale_options.keys()), key="batch_colormap")

    def render_3d_preview(X, Y, Z, colormap):
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=colormap)])
        fig.update_layout(
            scene=dict(zaxis_title='Depth', xaxis_title='X', yaxis_title='Y'),
            margin=dict(l=0, r=0, b=0, t=0), height=600
        )
        return fig

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"📂 {uploaded_file.name}")
            image = Image.open(uploaded_file).convert("L")
            img_array = np.array(image)
            img_array = np.flipud(img_array)

            height_map = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * depth
            X = np.linspace(0, width, img_array.shape[1])
            Y = np.linspace(0, height, img_array.shape[0])
            X, Y = np.meshgrid(X, Y)
            Z = height_map

            col_img, col_map = st.columns(2)
            with col_img:
                st.image(image, caption="Grayscale Input Image", use_column_width=True)
            with col_map:
                st.image(height_map, caption="Heightmap Visualization", use_column_width=True, clamp=True)

            fig = render_3d_preview(X, Y, Z, colorscale_options[selected_colorscale])
            st.plotly_chart(fig, use_container_width=True)

            stl_data = []
            for i in range(Z.shape[0] - 1):   
                for j in range(Z.shape[1] - 1):
                    p1 = [X[i][j], Y[i][j], Z[i][j]]
                    p2 = [X[i][j + 1], Y[i][j + 1], Z[i][j + 1]]
                    p3 = [X[i + 1][j], Y[i + 1][j], Z[i + 1][j]]
                    p4 = [X[i + 1][j + 1], Y[i + 1][j + 1], Z[i + 1][j + 1]]
                    stl_data.append([p1, p2, p3])
                    stl_data.append([p3, p2, p4])

            stl_array = np.zeros(len(stl_data), dtype=stl_mesh.Mesh.dtype)
            for i, f in enumerate(stl_data):
                stl_array["vectors"][i] = np.array(f)

            model_mesh = stl_mesh.Mesh(stl_array)
            stl_filename = f"{os.path.splitext(uploaded_file.name)[0]}.stl"
            stl_path = os.path.join(tempfile.gettempdir(), stl_filename)
            model_mesh.save(stl_path)

            with open(stl_path, "rb") as f:
                st.download_button(f"💾 Download STL for {uploaded_file.name}", f, file_name=stl_filename)

            os.remove(stl_path)


from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


with tab5:
    st.header("📄 Relief Report")
    if 'relief' in st.session_state:
        relief = st.session_state['relief']

        st.subheader("📷 Original Image")
        if 'original_image' in st.session_state:
            st.image(st.session_state['original_image'], caption="Original Uploaded Image")
            original_img_path = os.path.join(tempfile.gettempdir(), "original_image_temp.png")
            st.session_state['original_image'].save(original_img_path)

        st.subheader("🗺️ Grayscale Height Map")
        fig, ax = plt.subplots()
        ax.imshow(relief, cmap='gray')
        ax.axis('off')
        ax.set_title("Relief Height Map")
        heightmap_path = os.path.join(tempfile.gettempdir(), "heightmap_temp.png")
        fig.savefig(heightmap_path, bbox_inches='tight', dpi=150)
        st.pyplot(fig)

        st.subheader("📐 Relief Dimensions")
        min_h = np.min(relief)
        max_h = np.max(relief)
        avg_h = np.mean(relief)
        stats_text = f"""
Relief Dimensions:
Rows x Columns: {relief.shape[0]} x {relief.shape[1]}
Min Height: {min_h:.2f} mm
Max Height: {max_h:.2f} mm
Average Height: {avg_h:.2f} mm
"""
        st.text(stats_text)

        st.subheader("📊 Relief Height Histogram")
        fig2, ax2 = plt.subplots()
        ax2.hist(relief.flatten(), bins=50, color='skyblue', edgecolor='black')
        ax2.set_title("Height Distribution")
        histogram_path = os.path.join(tempfile.gettempdir(), "histogram_temp.png")
        fig2.savefig(histogram_path, bbox_inches='tight', dpi=150)
        st.pyplot(fig2)

        gcode_text = st.session_state.get('gcode', None)
        if gcode_text:
            st.subheader("🖨️ G-code Preview")
            st.code(gcode_text, language='gcode')
            gcode_lines = gcode_text.strip().split('\n')
        else:
            gcode_lines = []

        # === PDF GENERATION ===
        pdf_path = os.path.join(tempfile.gettempdir(), "relief_report.pdf")
        c = canvas.Canvas(pdf_path, pagesize=A4)
        pdf_width, pdf_height = A4
        margin = 40
        y = pdf_height - margin

        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, y, "Relief Report")
        y -= 30

        # Stats block
        c.setFont("Helvetica", 12)
        for line in stats_text.strip().split("\n"):
            c.drawString(margin, y, line.strip())
            y -= 18

        # Add images compactly
        if os.path.exists(original_img_path):
            y -= 10
            c.drawImage(ImageReader(original_img_path), margin, y - 120, width=180, height=120, mask='auto')
            c.drawString(margin, y - 130, "Original Image")

        if os.path.exists(heightmap_path):
            c.drawImage(ImageReader(heightmap_path), margin + 200, y - 120, width=180, height=120, mask='auto')
            c.drawString(margin + 200, y - 130, "Height Map")

        y -= 160

        if os.path.exists(histogram_path):
            c.drawImage(ImageReader(histogram_path), margin, y - 150, width=360, height=150, mask='auto')
            c.drawString(margin, y - 160, "Relief Height Histogram")
            y -= 180

        # Add G-code Preview (up to 40 lines max)
        if gcode_lines:
            c.setFont("Courier", 8)
            c.drawString(margin, y, "G-code Preview (First 40 lines):")
            y -= 14
            for line in gcode_lines[:40]:
                if y < 50:
                    break
                c.drawString(margin, y, line)
                y -= 10

        c.save()

        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Relief Report (PDF)", f, file_name="relief_report.pdf", mime="application/pdf")

        st.success("✅ Report generated as a single-page PDF.")
    else:
        st.warning("⚠️ Please generate relief from Tab 1 first.")
