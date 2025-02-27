import os

try:
    import cv2
except ImportError:
    os.system("pip install opencv-python-headless")
    import cv2  # Try importing again after installation
    
import asyncio
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import sys
import asyncio

# Fix event loop issue in Streamlit Cloud (PyTorch)
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    pass

# Ensure the script's directory is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Now try to import chess_utils
try:
    import chess_utils
    st.success("✅ Successfully imported chess_utils!")
except ModuleNotFoundError as e:
    st.error(f"⚠ Import Error: {e}")
    st.stop()

# Load YOLO models (using your best models)
st.write("🔄 Loading YOLO models...")
try:
    board_model = YOLO("runs/detect_board/train/weights/best.pt")  # Best board detection model
    piece_model = YOLO("runs/detect_pieces/train/weights/best.pt")  # Best piece detection model
    st.success("✅ Models loaded successfully!")
except FileNotFoundError as e:
    st.error(f"⚠ Model file missing: {e}")
    st.stop()

# Upload Image
uploaded_file = st.file_uploader("📂 Upload a Chessboard Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV image format
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert to NumPy array for OpenCV processing

    # Ensure 3-channel RGB format
    if image.ndim == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # **Step 1: Detect the Chessboard**
    st.write("🔄 Detecting chessboard...")
    try:
        board_results = board_model(image, iou=0.1)[0]  # Lower IoU to avoid duplicate detections
    except Exception as e:
        st.error(f"⚠ Error detecting chessboard: {e}")
        st.stop()

    # Extract crossings (detected board intersections)
    if len(board_results.boxes) == 0:
        st.error("⚠ No chessboard detected. Try another image.")
        st.stop()
    else:
        crossings = []
        for box in board_results.boxes.data:
            x_min, y_min, x_max, y_max, conf, cls = box.cpu().numpy()
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            crossings.append((int(x_center), int(y_center)))

        # Convert to NumPy array
        crossings = np.array(crossings)

        # **Step 2: Structure Crossings into an 8×8 Grid**
        structured_grid = chess_utils.complete_grid(crossings, image.shape)


        # **Step 3: Generate Grid and Draw Infinite Lines**
        grid = chess_utils.complete_grid(structured_grid, image.shape)
        image_with_grid, horizontal_lines, vertical_lines = chess_utils.draw_infinite_grid(image, grid)

        # Display the result
        st.image(image_with_grid, caption="🟩 Detected Chessboard Grid", use_column_width=True)

    # **Step 4: Detect Chess Pieces**
    st.write("🔄 Detecting chess pieces...")
    try:
        piece_results = piece_model(image, iou=0.5, conf=0.32)[0]
    except Exception as e:
        st.error(f"⚠ Error detecting chess pieces: {e}")
        st.stop()

    if len(piece_results.boxes) == 0:
        st.error("⚠ No chess pieces detected.")
        st.stop()
    else:
        st.success("✅ Chess pieces detected!")

        # Extract detected pieces
        piece_names = piece_model.model.names
        pieces = []
        for box in piece_results.boxes.data:
            x_min, y_min, x_max, y_max, conf, cls = box.cpu().numpy()
            piece_center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
            class_label = int(cls)
            piece_name = piece_names[class_label]
            pieces.append((piece_center, piece_name))

        # **Step 5: Convert to Chessboard Representation**
        board_df = chess_utils.create_chessboard_dataframe(pieces, horizontal_lines, vertical_lines)
        board_df = chess_utils.reorient_board(board_df)  # Ensure proper orientation
        board_df = chess_utils.ensure_kings(board_df)  # Ensure at least one king exists

        # **Step 6: Convert to FEN Notation**
        fen_string = chess_utils.df_to_fen(board_df)
        st.write("📜 **FEN Representation:**")
        st.code(fen_string, language="text")

        # **Step 7: Display the Chessboard**
        import chess
        import chess.svg
        from IPython.display import display, SVG

        board = chess.Board(fen_string)
        svg_board = chess.svg.board(board, size=500)
        st.write("♟ **Chessboard Visualization:**")
        st.image(chess_utils.render_svg(svg_board), caption="♟ Chessboard Position", use_column_width=True)

    st.write("✅ **Processing Complete!**")
