# 🏁 Chess Vision AI 🏁

This project detects, recognizes, and evaluates chess positions from images using **YOLO-based board and piece detection**, **Stockfish engine analysis**, and **python-chess visualization**.

## 🚀 Features

- **Chessboard Detection (YOLOv8)**
  - Identifies board intersections and structures the grid.
  - Extends an infinite grid to ensure robustness.

- **Piece Recognition (YOLOv8)**
  - Detects and assigns chess pieces to board squares.

- **Position Reconstruction**
  - Converts detected pieces into **Forsyth-Edwards Notation (FEN)**.
  - Ensures correct board orientation (White at the bottom).

- **Stockfish Evaluation**
  - Finds the **best move** for White or Black.
  - Provides **positional evaluation metrics** (material, mobility, king safety, etc.).
  - Displays an interactive chessboard with recommended moves.

## 📦 Installation

### **1️⃣ Install Dependencies**
Install the required packages:

```bash
pip install -r requirements.txt
```

### **2️⃣ Install Stockfish (Required for Move Evaluation)**
Download and extract **Stockfish 15+**:
- **Windows**: [Download](https://stockfishchess.org/download/) → Add `stockfish.exe` to PATH.
- **Linux**: Install via package manager:

## 🛠 Usage

### **Run the Chess Vision Pipeline**
To detect the board and evaluate the position, run:

```bash
python chess_vision.py
```

### **Train Custom YOLO Models (Optional)**
To retrain the board and piece detection models:

```python
# Uncomment in chess_vision.py
train_board_detection()
train_piece_detection()
```

### **Interactive Analysis in Jupyter Notebook**
To analyze images interactively:

```python
from chess_utils import display_eval, get_best_move

best_move_white = get_best_move(full_fen_w)
print("Best move for white:", best_move_white)
display_eval(full_fen_w, best_move_white)
```

## 📊 Chess Evaluation Metrics

| **Metric**       | **Description** |
|------------------|------------------------------------------------|
| **Material**     | Overall material balance (e.g., pawn=1, queen=9). |
| **Imbalance**    | Positional differences beyond raw material count. |
| **Pawns**        | Pawn structure analysis (passed, isolated, doubled). |
| **Knights**      | Activity of knights (outposts, mobility). |
| **Bishops**      | Bishop activity, control over long diagonals. |
| **Rooks**        | Rook activity (open files, 7th rank control). |
| **Queens**       | Queen's mobility and placement effectiveness. |
| **Mobility**     | Legal move count, indicating piece activity. |
| **King Safety**  | King protection (pawn cover, exposure). |
| **Threats**      | Tactical threats present in the position. |
| **Passed Pawns** | Presence and strength of passed pawns. |
| **Space**        | Board control and territory. |
| **Winnability**  | Heuristic measure of winning chances. |
| **Total**        | Aggregate evaluation (in centipawns). |

## 🏗 Project Structure

```plaintext
📂 Chess-Vision-AI
│── chess_vision.py           # Main script for board & piece detection
│── stockfish_utils.py        # Functions for move evaluation
│── chess_utils.py            # Grid detection, FEN conversion, & visualization
│── models.py                 # Functions to train YOLO models
│── runs/                     # YOLO model weights
│── data/                     # Sample chess images
│── requirements.txt          # Dependencies
│── README.md                 # This file
```

## 📌 Example Output

### **Detected Chessboard**
![Detected Board](https://user-images.githubusercontent.com/example/detected_board.jpg)

### **Stockfish Best Move**
```plaintext
Best move for White: e2e4
Evaluation: +0.78 (White slightly better)
```

### **Visualized Board with Best Move**
![Best Move](https://user-images.githubusercontent.com/example/best_move.svg)

## 🔗 Resources
- [Stockfish's website](https://stockfishchess.org/)
- [Stockfish 15 Chess Engine download link](https://drive.google.com/drive/folders/1ASj7nGkFlZB-RLZxcmYa47sq4moN4aAb)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [python-chess](https://python-chess.readthedocs.io/en/latest/)

## 📜 License
This project is **MIT Licensed** – feel free to use, modify, and distribute.

---

🔥 **Contributors Welcome!** Submit pull requests or issues to improve the pipeline. 🚀♟️
