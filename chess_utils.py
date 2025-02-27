import cv2

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

from scipy.interpolate import griddata

def complete_grid(crossings, image_shape):
    """
    Ensures that the detected chessboard intersections form a complete 7x7 grid.
    If some crossings are missing, they are estimated using interpolation.

    Steps:
    1. Sort detected crossings by Y (top to bottom), then by X (left to right).
    2. Attempt to reshape them into a 7x7 grid.
    3. If reshaping fails, apply K-Means clustering to group crossings into 7 rows.
    4. Within each row, sort points by X-coordinate and ensure exactly 7 per row.
    5. Fill missing values using interpolation from neighboring points.
    """
    
    # Convert to numpy array
    crossings = np.array(crossings)
    
    # If all 49 crossings are detected, reshape directly
    if len(crossings) == 49:
        crossings = sorted(crossings, key=lambda p: (p[1], p[0]))  # Sort by Y, then X
        grid = np.array(crossings).reshape(7, 7, 2)
        return grid

    print(f"Warning: Detected only {len(crossings)} points. Estimating missing crossings...")

    # Use K-Means to cluster into 7 row groups (using Y-coordinates)
    kmeans = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(crossings[:, 1].reshape(-1, 1))
    row_labels = kmeans.labels_

    # Sort by row clusters, then by X within each row
    sorted_crossings = []
    for i in range(7):
        row_points = crossings[row_labels == i]  # Select only points in the row
        row_points = sorted(row_points, key=lambda p: p[0])  # Sort row by X
        sorted_crossings.append(row_points)

    # Ensure each row has exactly 7 points (interpolate if necessary)
    grid = np.full((7, 7, 2), np.nan)  # Initialize grid with NaNs
    for i, row in enumerate(sorted_crossings):
        row = np.array(row)
        if row.shape[0] != 7:
            x_vals = row[:, 0]
            y_vals = row[:, 1]
            x_new = np.linspace(x_vals.min(), x_vals.max(), 7)  # Evenly space missing points
            y_new = np.interp(x_new, x_vals, y_vals)  # Interpolate Y values for new x's
            row = np.column_stack((x_new, y_new))  # Reconstruct row as 7 (x,y) pairs
        grid[i] = row  # Assign row to grid

    # Interpolate any remaining NaNs in the grid using griddata
    known_points = np.array([
        (i, j, grid[i, j][0], grid[i, j][1])
        for i in range(7) for j in range(7)
        if not np.isnan(grid[i, j]).any()
    ])
    missing_points = np.array([
        (i, j)
        for i in range(7) for j in range(7)
        if np.isnan(grid[i, j]).any()
    ])

    if missing_points.size > 0 and known_points.size > 0:
        known_coords = known_points[:, :2]
        known_values = known_points[:, 2:]
        estimated_values = griddata(known_coords, known_values, missing_points, method='cubic')
        for (i, j), (x, y) in zip(missing_points, estimated_values):
            grid[i, j] = [x, y]

    return grid




def compute_line_equation(p1, p2, img_shape):
    """
    Computes the infinite line equation given two points and extends it to image boundaries.

    Steps:
    1. Determine if the line is vertical (special case handling).
    2. Compute the slope and y-intercept.
    3. Compute the intersections of this line with the image boundaries.
    4. Return only valid points within the image dimensions.
    """

    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    
    if x1 == x2:  # Handle vertical lines
        return (x1, 0), (x1, img_shape[0] - 1)

    # Compute slope and intercept
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Compute intersections with image borders
    x_min, x_max = 0, img_shape[1] - 1
    y_min, y_max = 0, img_shape[0] - 1

    # Compute extended points
    y_xmin = int(slope * x_min + intercept)  # Left boundary
    y_xmax = int(slope * x_max + intercept)  # Right boundary

    if slope == 0:
        slope = 1e-6
    x_ymin = int((y_min - intercept) / slope)  # Top boundary
    x_ymax = int((y_max - intercept) / slope)  # Bottom boundary

    # Clip to image bounds
    points = [
        (x_min, y_xmin),
        (x_max, y_xmax),
        (x_ymin, y_min),
        (x_ymax, y_max)
    ]
    
    # Keep only points within image bounds
    valid_points = [(int(x), int(y)) for x, y in points if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]]

    if len(valid_points) < 2:
        return (x1, y1), (x2, y2)  # If no valid extensions, return original points

    return valid_points[0], valid_points[-1]


def draw_infinite_grid(image, grid):
    """
    Draws an extended chessboard grid using detected intersections.

    Steps:
    1. Extend each detected horizontal line across the entire image width.
    2. Extend each detected vertical line across the entire image height.
    3. Draw the extended grid on the image.
    """

    img_h, img_w, _ = image.shape
    
    horizontal_lines = []
    vertical_lines = []

    # Extend and draw horizontal lines
    for row in grid:
        p1, p2 = row[0], row[-1]
        p1_ext, p2_ext = compute_line_equation(p1, p2, image.shape)
        horizontal_lines.append(np.array([p1_ext, p2_ext]))
        cv2.line(image, p1_ext, p2_ext, (0, 255, 0), 2)

    # Extend and draw vertical lines
    for col in range(grid.shape[1]):
        p1, p2 = grid[0, col], grid[-1, col]
        p1_ext, p2_ext = compute_line_equation(p1, p2, image.shape)
        vertical_lines.append(np.array([p1_ext, p2_ext]))
        cv2.line(image, p1_ext, p2_ext, (0, 255, 0), 2)

    return image, horizontal_lines, vertical_lines


def intersect_horizontal_line_with_line(grid_line, y_val):
    """
    Computes the x-coordinate where a given horizontal line (y = y_val) 
    intersects a vertical or diagonal grid line.

    Steps:
    1. Extract the endpoints of the grid line.
    2. Handle special cases where the grid line is nearly vertical (returns a fixed x).
    3. Compute the slope and y-intercept of the line.
    4. Use the line equation to solve for the x-coordinate at y = y_val.
    5. If the grid line is nearly horizontal, return None (ambiguous intersection).
    """

    (x1, y1), (x2, y2) = grid_line

    # Handle nearly vertical lines (x is constant)
    if abs(x2 - x1) < 1e-6:
        return x1  

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # If the grid line is nearly horizontal, intersection is undefined
    if abs(slope) < 1e-6:
        return None

    # Compute the x-coordinate of the intersection
    x_intersect = (y_val - intercept) / slope
    return x_intersect


def intersect_vertical_line_with_line(grid_line, x_val):
    """
    Computes the y-coordinate where a given vertical line (x = x_val) 
    intersects a horizontal or diagonal grid line.

    Steps:
    1. Extract the endpoints of the grid line.
    2. Handle special cases where the grid line is nearly horizontal (returns a fixed y).
    3. Compute the slope and y-intercept of the line.
    4. Use the line equation to solve for the y-coordinate at x = x_val.
    5. If the grid line is nearly vertical, return None (ambiguous intersection).
    """

    (x1, y1), (x2, y2) = grid_line

    # Handle nearly horizontal lines (y is constant)
    if abs(y2 - y1) < 1e-6:
        return y1  

    # If the grid line is nearly vertical, intersection is undefined
    if abs(x2 - x1) < 1e-6:
        return None

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Compute the y-coordinate of the intersection
    y_intersect = slope * x_val + intercept
    return y_intersect


def count_vertical_boundaries(piece_center, vertical_lines):
    """
    Determines how many vertical grid lines a horizontal line through the piece’s center crosses.

    Steps:
    1. Extract the x and y coordinates of the piece center.
    2. Iterate over all detected vertical grid lines.
    3. Compute the intersection of the horizontal line at y = piece_center[1] with each vertical grid line.
    4. Count how many of these intersections have x-coordinates smaller than the piece’s x-coordinate.
    5. This count represents how many vertical boundaries exist to the left of the piece, 
       which corresponds to its column index on the chessboard.
    """

    x_piece, y_piece = piece_center
    count = 0

    for line in vertical_lines:
        x_int = intersect_horizontal_line_with_line(line, y_piece)
        if x_int is not None and x_piece > x_int:
            count += 1

    return count


def count_horizontal_boundaries(piece_center, horizontal_lines):
    """
    Determines how many horizontal grid lines a vertical line through the piece’s center crosses.

    Steps:
    1. Extract the x and y coordinates of the piece center.
    2. Iterate over all detected horizontal grid lines.
    3. Compute the intersection of the vertical line at x = piece_center[0] with each horizontal grid line.
    4. Count how many of these intersections have y-coordinates larger than the piece’s y-coordinate.
    5. This count represents how many horizontal boundaries exist above the piece, 
       which corresponds to its row index on the chessboard.
    """

    x_piece, y_piece = piece_center
    count = 0

    for line in horizontal_lines:
        y_int = intersect_vertical_line_with_line(line, x_piece)
        if y_int is not None and y_piece > y_int:
            count += 1

    return count


def flexible_priority_assignments(square_assignments):
    """
    Resolves conflicts in square assignments by prioritizing piece placement.

    This function handles cases where multiple pieces are detected on the same square
    and applies the following rules:
    
    1. **If there is no conflict** (only one piece assigned to a square), it is kept.
    2. **If multiple pieces are detected on the same square:**
        - Kings are deprioritized if there are non-king candidates.
        - If a piece type has not yet been placed anywhere on the board, it gets priority.
        - If all piece types are already placed, the one with the highest predefined 
          priority (lowest value in `flexible_priority`) is chosen.
        - Among flexible pieces, bishops are preferred over queens, followed by rooks 
          and knights.
    
    Parameters:
        square_assignments (dict): A dictionary where keys are square names (e.g., "e4"),
                                   and values are lists of tuples representing assigned pieces.
                                   Each tuple contains (piece_name, piece_center).
    
    Returns:
        dict: A dictionary of final resolved assignments where each square contains 
              at most one piece after applying priority-based conflict resolution.
    """

    # Define flexible priorities (lower number = higher priority)
    # Bishops have the highest priority (1), then queens (2), rooks (3), and knights (4).
    flexible_priority = {
        "white-bishop": 1, "black-bishop": 1,
        "white-queen": 2,  "black-queen": 2,
        "white-rook": 3,   "black-rook": 3,
        "white-knight": 4, "black-knight": 4
    }

    # Track how many flexible pieces have been placed to ensure variety in piece types.
    global_flexible_counts = {piece: 0 for piece in flexible_priority}

    final_assignments = {}

    for square, assigned_pieces in square_assignments.items():
        if len(assigned_pieces) == 1:
            # No conflict: directly assign the single piece.
            chosen = assigned_pieces[0]
            final_assignments[square] = chosen
            piece_type = chosen[0]
            if piece_type in flexible_priority:
                global_flexible_counts[piece_type] += 1
        else:
            # Conflict: multiple pieces detected on the same square.
            # First, exclude kings if non-king candidates exist.
            non_king_candidates = [p for p in assigned_pieces if p[0] not in ["white-king", "black-king"]]
            candidates = non_king_candidates if non_king_candidates else assigned_pieces  # Keep all if only kings exist.

            # Prioritize placing pieces that haven't been assigned anywhere yet.
            candidates_not_on_board = [
                p for p in candidates 
                if p[0] in flexible_priority and global_flexible_counts[p[0]] == 0
            ]
            if candidates_not_on_board:
                # Choose the one with the highest priority (lowest flexible_priority value).
                chosen = min(candidates_not_on_board, key=lambda p: flexible_priority.get(p[0], 999))
            else:
                # Otherwise, choose the one with the highest predefined priority.
                chosen = min(candidates, key=lambda p: flexible_priority.get(p[0], 999))

            final_assignments[square] = chosen
            piece_type = chosen[0]
            if piece_type in flexible_priority:
                global_flexible_counts[piece_type] += 1

            print(f"Conflict at {square}: {assigned_pieces} → Keeping {chosen}")

    return final_assignments


def ensure_kings(chessboard):
    """
    Ensures there is at least one white and one black king on the board.
    If a king is missing, it replaces a bishop ('white-bishop' or 'black-bishop') with a king.
    """
    # Flatten the board to check for missing kings
    flat_board = chessboard.flatten().tolist()
    white_king_missing = "white-king" not in flat_board
    black_king_missing = "black-king" not in flat_board
    
    
    # If both kings are present, no modification needed
    if not white_king_missing and not black_king_missing:
        
        return chessboard

    for row in range(8):
        for col in range(8):
            piece = chessboard[row, col]
            
            # Replace a white bishop with a king if the white king is missing
            if white_king_missing and piece == "white-bishop":
                chessboard[row, col] = "white-king"
                print(f"White king was missing. Transformed 'white-bishop' at ({row}, {col}) into 'white-king'.")
                white_king_missing = False  # Stop replacing

            # Replace a black bishop with a king if the black king is missing
            elif black_king_missing and piece == "black-bishop":
                chessboard[row, col] = "black-king"
                print(f"Black king was missing. Transformed 'black-bishop' at ({row}, {col}) into 'black-king'.")
                black_king_missing = False  # Stop replacing

            # If both replacements are done, exit early
            if not white_king_missing and not black_king_missing:
                break

    

    return chessboard  # Return updated board


def reorient_board(board):
    """
    Reorients an 8x8 chessboard (as a NumPy matrix) so that White pieces are at the bottom (higher row indices)
    and Black pieces are at the top (lower row indices).

    The function tests all 4 possible rotations (0°, 90°, 180°, 270°) and selects the one where the white pieces
    are positioned as far down as possible relative to the black pieces. It does this by computing, for each rotation,
    the difference between the minimum row index of all White pieces and the maximum row index of all Black pieces.
    (Rows are numbered 0 (top) to 7 (bottom).)

    A higher score (white_min - black_max) indicates that the white pieces are lower on the board and the black pieces
    are higher, which is the desired orientation.

    Args:
        board (np.ndarray): An 8x8 NumPy array representing the chessboard, where each cell is a string 
                            (e.g., "white-pawn", "black-king", or "" for empty).

    Returns:
        np.ndarray: The reoriented 8x8 board with White on the bottom and Black on the top.
    """

    # Generate all 4 possible rotations of the board.
    rotations = {
        "0": board,  # no rotation
        "90": np.rot90(board, k=3),  # 90° clockwise (rot90 with k=3)
        "180": np.rot90(board, k=2), # 180° rotation
        "270": np.rot90(board, k=1)  # 270° clockwise (or 90° counterclockwise)
    }

    best_score = -1000  # Initialize with a very low score.
    best_key = None

    # Evaluate each rotation using a heuristic:
    # We compute:
    #   white_min = minimum row index among all cells containing a white piece,
    #   black_max = maximum row index among all cells containing a black piece.
    # The score is defined as: white_min - black_max.
    # A higher score means white pieces are further down (closer to row 7) and black pieces are further up (closer to row 0).
    for key, arr in rotations.items():
        white_rows = []
        black_rows = []
        for r in range(8):
            for c in range(8):
                piece = arr[r, c]
                if piece.startswith("white"):
                    white_rows.append(r)
                elif piece.startswith("black"):
                    black_rows.append(r)
        # Only compute score if we found both white and black pieces.
        if white_rows and black_rows:
            white_min = min(white_rows)
            black_max = max(black_rows)
            score = white_min - black_max
        else:
            score = -1000  # If one color is missing, this rotation is not optimal.
        if score > best_score:
            best_score = score
            best_key = key

    best_array = rotations[best_key]
    print(f"Chosen rotation: {best_key}° (score = {best_score})")
    return best_array


def df_to_fen(df):
    """
    Converts a DataFrame representation of a chessboard to a FEN position string.

    Steps:
    1. Define a mapping from detected piece names to FEN notation.
    2. Flip the DataFrame vertically to match the FEN row order.
    3. Iterate through each row, replacing empty squares with numbers and pieces with FEN characters.
    4. Join rows with "/" to create the final FEN string.
    """

    mapping = {
        "white-king": "K",
        "white-queen": "Q",
        "white-rook": "R",
        "white-bishop": "B",
        "white-knight": "N",
        "white-pawn": "P",
        "black-king": "k",
        "black-queen": "q",
        "black-rook": "r",
        "black-bishop": "b",
        "black-knight": "n",
        "black-pawn": "p",
    }
    
    fen_rows = []

    for rank in df.index:
        row_str = ""
        empty_count = 0
        for file in df.columns:
            piece = df.loc[rank, file]
            if piece == "" or pd.isna(piece):
                empty_count += 1
            else:
                if empty_count:
                    row_str += str(empty_count)
                    empty_count = 0
                fen_piece = mapping.get(piece.lower(), piece)
                row_str += fen_piece
        if empty_count:
            row_str += str(empty_count)
        fen_rows.append(row_str)
        
    return "/".join(fen_rows)


if __name__ == "__main__":
    """
    Main entry point for the script. This module defines utility functions for detecting,
    processing, and visualizing chessboards using computer vision.
    """
    print("This is a file containing utility functions for the chess vision project.")
