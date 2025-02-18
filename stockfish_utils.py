import os
import gc
import subprocess
import threading
import queue
from IPython.display import display, HTML
import chess
import pandas as pd
import numpy as np
import re

if os.name == 'nt':
    STOCKFISH_15_PATH = r"stockfish_15_windows\stockfish-windows-2022-x86-64-avx2.exe"

elif os.name == 'posix':
    STOCKFISH_15_PATH = r"stockfish_15_linux/stockfish-ubuntu-20.04-x86-64-avx2"


def blend_mg_eg_classical(df):
    # Iterate over each sub-column in the 'MG' group
    for subcol in df['MG'].columns:
        # Check that a corresponding column exists in the 'EG' group
        if subcol in df['EG'].columns:
            # Create a new 'Blended' column for this sub-column.
            # Here we assume that calculate_phase_ratio takes a phase (i.e. row index) as input.
            df[('Blended', subcol)] = df.apply(
                lambda row: row[('MG', subcol)] * calculate_phase_ratio(row.name) +
                            row[('EG', subcol)] * (1 - calculate_phase_ratio(row.name)),
                axis=1
            )
    return df["Blended"]


# Function to calculate the phase ratio
def calculate_phase_ratio(fen):
    # Piece values for phase calculation
    piece_phase_values = {
        'P': 0, 'p': 0,   # Pawns are not counted for phase
        'N': 1, 'n': 1,
        'B': 1, 'b': 1,
        'R': 2, 'r': 2,
        'Q': 4, 'q': 4,
        'K': 0, 'k': 0   # Kings are not counted for phase
    }

    # Maximum phase (all non-pawn pieces present)
    MAX_PHASE = 24

    board_part = fen.split(' ')[0]
    current_phase = sum(piece_phase_values.get(char, 0) for char in board_part if char.isalpha())
    return current_phase / MAX_PHASE if MAX_PHASE else 0


def generate_eval_bar_html(eval_scores, titles):
    """
    Generate HTML for evaluation bars.
    
    Args:
        eval_scores (list of floats): Evaluation scores, where positive means more white and negative means more black.
        titles (list of str): Titles for each evaluation bar.

    Returns:
        str: HTML for rendering the evaluation bars.
    """
    bars = []
    for score, title in zip(eval_scores, titles):
        # Calculate white height (in percentage)
        white_height = 50 + (score / 5) * 50
        white_height = max(5, min(95, white_height))  # Clamp to [5, 95]
        
        # Determine text position and color
        if score < 0:
            text_position = "top: 5px;"  # Slight offset from the top
            text_color = "white"
        else:
            text_position = "bottom: 5px;"  # Slight offset from the bottom
            text_color = "black"
        
        # Create the individual bar
        bars.append(f"""
        <div style="flex: 1; height: 400px; border: 1px solid #ccc; position: relative; background: black; text-align: center; display: flex; flex-direction: column; justify-content: flex-end;">
            <!-- White section -->
            <div style="position: absolute; bottom: 0; height: {white_height}%; background: white; width: 100%;"></div>
            <!-- Gray dashed line in the middle -->
            <div style="position: absolute; top: 50%; width: 100%; height: 0; border-top: 2px dashed gray;"></div>
            <!-- Score label -->
            <div style="position: absolute; {text_position} width: 100%; text-align: center; font-weight: bold; color: {text_color}; z-index: 2;">
                {score:+.2f}
            </div>
            <!-- Title -->
            <div style="position: absolute; bottom: -40px; width: 100%; text-align: center; font-size: 14px; color: white;">
                {title}
            </div>
        </div>
        """)

    return f"""
    <div style="display: flex; justify-content: space-around; align-items: flex-end; gap: 10px; margin-bottom: 50px;">
        {''.join(bars)}
    </div>
    """


def display_eval(fen, move):
    # Initialize chess board
    board = chess.Board(fen)
    if move != None:
        # Parse source and destination squares from the UCI move
        source_square = chess.parse_square(move[:2])
        dest_square = chess.parse_square(move[2:4])
        
        color = "blue"
        
        # Create an arrow from the source square to the destination square
        move_arrow = chess.svg.Arrow(source_square, dest_square, color=color)
        
        # Display the board with the arrow
        display(HTML(f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            {chess.svg.board(board=board, size=400, arrows=[move_arrow])}
        </div>
        """))
    else:
        # Display initial chess board
        display(HTML(f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            {chess.svg.board(board=board, size=400)}
        </div>
        """))

    # Evaluation
    # We evaluate that position
    classical_eval, _, _, _ = get_eval(STOCKFISH_15_PATH, fen)

    # Example usage with evaluation scores and titles
    eval_scores = blend_mg_eg_classical(classical_eval)["Total"]
    titles = blend_mg_eg_classical(classical_eval).index
    html_content = generate_eval_bar_html(eval_scores, titles)

    # Display the generated HTML
    display(HTML(html_content))
    del board
    gc.collect()


def get_eval(exe_path, custom_fen):
    """
    Runs Stockfish to evaluate a custom FEN position.

    Args:
        exe_path (str): Path to the Stockfish executable.
        custom_fen (str): The FEN string representing the chess position.

    Returns:
        tuple: Parsed DataFrames for classical evaluation, NNUE piece values, 
               NNUE network contributions, and final evaluations.
    """
    # Start Stockfish engine
    engine = subprocess.Popen(
        [exe_path],
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1  # Line-buffered output
    )

    # Get and apply options
    default_options = get_default_options(engine)
    set_options_as_commands(engine, default_options)

    # Queue to hold subprocess output
    output_queue = queue.Queue()

    # Start the thread to read stdout
    thread = threading.Thread(target=enqueue_output, args=(engine.stdout, output_queue))
    thread.daemon = True
    thread.start()

    # Initialize Stockfish in UCI mode
    send_command(engine, "uci")
    read_output_until("uciok", output_queue)  # Wait for "uciok" signal


    # Set custom position
    send_command(engine, f"position fen {custom_fen}")

    # evaluate
    send_command(engine, "eval")

    trace_output = read_output_until("Final evaluation", output_queue)  # Wait until "Final evaluation" appears
    
    # Clean up
    engine.stdin.close()
    engine.stdout.close()
    engine.terminate()
    thread.join()
    
    if "Final evaluation: none (in check)" in trace_output:
        return invalid_trace()

    return parse_and_save_output(trace_output)


def invalid_trace():
    """
    Initializes and returns multiple DataFrames filled with None.
    This is for the cases when the position cannot be analyzed (such as King in check)

    """
    df1 = pd.DataFrame(np.nan, index=[
        "Material", "Imbalance", "Pawns", "Knights", "Bishops", 
        "Rooks", "Queens", "Mobility", "King safety", "Threats", 
        "Passed", "Space", "Winnable", "Total"
    ], columns=pd.MultiIndex.from_tuples([
        ("MG", "White"), 
        ("MG", "Black"), 
        ("MG", "Total"), 
        ("EG", "White"), 
        ("EG", "Black"), 
        ("EG", "Total")
    ]))
    
    df2 = pd.DataFrame(None, index=[8, 7, 6, 5, 4, 3, 2, 1], columns=["a", "b", "c", "d", "e", "f", "g", "h"])
    
    # Initialize the DataFrame with None
    df3 = pd.DataFrame(None, index=range(8), columns=["PSQT", "Positional", "Total", "Complexity", "Used"])
    df3["Used"] = 0
    df3.loc[-1, "Used"] = 1
    
    df4 = pd.DataFrame(None, index=["Classical evaluation", "NNUE evaluation", "Final evaluation"], columns=["Value"])
    
    return df1, df2, df3, df4


def parse_nnue_derived_piece_values_to_board_with_tuples(nnue_piece_values_lines):
    """
    Parses NNUE-derived piece values and organizes them into a DataFrame resembling a chess board.

    Args:
        nnue_piece_values_lines (list): Lines containing NNUE piece values.

    Returns:
        pd.DataFrame: A DataFrame representing the chessboard with piece values.
    """
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = list(range(8, 0, -1))  # Ranks from 8 to 1

    # Initialize an empty board DataFrame
    board = pd.DataFrame(index=ranks, columns=files, dtype=object)

    # Filter valid lines (exclude borders and irrelevant lines)
    valid_lines = [line for line in nnue_piece_values_lines if "|" in line and not line.startswith("+")]

    # Ensure valid lines come in pairs: one for pieces, one for values
    if len(valid_lines) % 2 != 0:
        raise ValueError("Mismatched piece-value lines. Ensure the input format is correct.")

    rank_index = 8  # Start from rank 8
    for i in range(0, len(valid_lines), 2):
        pieces_line = valid_lines[i].strip('|').split('|')
        values_line = valid_lines[i + 1].strip('|').split('|')

        if len(pieces_line) != 8 or len(values_line) != 8:
            print(f"Warning: Line pair at index {i} does not have 8 entries. Skipping.")
            continue

        for file_index, file in enumerate(files):
            piece = pieces_line[file_index].strip()
            value = values_line[file_index].strip()

            # Determine the tuple value for the cell
            if piece == '' and value == '':
                cell_value = None  # Empty cell
            elif piece == 'K':  # White king
                cell_value = (piece, float('inf'))
            elif piece == 'k':  # Black king
                cell_value = (piece, float('-inf'))
            else:
                try:
                    cell_value = (piece, float(value)) if value else (piece, None)
                except ValueError:
                    cell_value = (piece, None)

            # Assign the parsed value to the board
            board.at[rank_index, file] = cell_value

        rank_index -= 1  # Move to the next rank

    return board


def parse_nnue_network_contributions(nnue_contributions_lines):
    """
    Parses NNUE network contributions into a DataFrame, including a binary column to track 
    whether a bucket is marked as used.

    Args:
        nnue_contributions_lines (list): Lines containing NNUE network contributions.

    Returns:
        pd.DataFrame: A DataFrame with NNUE contributions by bucket and a 'Used' column.
    """
    rows = []
    for line in nnue_contributions_lines:
        # Skip empty lines and separator lines
        if not line.strip() or line.strip().startswith("+"):
            continue

        # Check if the line has the arrow indicating "this bucket is used"
        is_used = "<-- this bucket is used" in line
        line = line.replace("<-- this bucket is used", "").strip()  # Clean up the line

        # Split the line into parts
        parts = line.strip('|').split('|')
        if len(parts) < 4:
            print(f"Skipping invalid line: {line}")
            continue
        
        try:
            bucket, material, positional, total = [p.strip() for p in parts[:4]]

            # Parse numeric values
            bucket = int(bucket)
            material = float(material.replace(" ", "").replace("+", ""))
            positional = float(positional.replace(" ", "").replace("+", ""))
            total = float(total.replace(" ", "").replace("+", ""))
            nnueComplexity = float(abs(material - positional))

            rows.append([bucket, material, positional, total, nnueComplexity, int(is_used)])
        except ValueError:
            continue

    # Create DataFrame with an additional 'Used' column
    return pd.DataFrame(
        rows, 
        columns=["Bucket", "PSQT", "Positional", "Total", "Complexity", "Used"]
    ).set_index("Bucket") if rows else pd.DataFrame(columns=["Bucket", "PSQT", "Positional", "Total", "Complexity", "Used"])


def parse_and_save_output(trace_output):
    """
    Parses different sections of Stockfish output dynamically.

    Args:
        trace_output (list): List of lines in the Stockfish trace output.

    Returns:
        tuple: Parsed DataFrames for classical evaluation, NNUE piece values, 
               NNUE network contributions, and final evaluations.
    """
    # Define markers for each section
    classical_eval_start = 'Contributing terms for the classical eval:'
    nnue_piece_values_start = "NNUE derived piece values"
    nnue_contributions_start = "NNUE network contributions"

    # Get indices dynamically
    piece_values_start, piece_values_end = get_section_indices(trace_output, nnue_piece_values_start, nnue_contributions_start)
    contributions_start, contributions_end = get_section_indices(trace_output, nnue_contributions_start, "")

    # Parse each section using the indices
    nnue_derived_piece_values = parse_nnue_derived_piece_values_to_board_with_tuples(trace_output[piece_values_start + 1 : piece_values_end])
    nnue_network_contributions = parse_nnue_network_contributions(trace_output[contributions_start + 1 : contributions_end])
    if classical_eval_start in trace_output:  # Classical evaluation is optional
        classical_values_start, classical_values_end = get_section_indices(trace_output, classical_eval_start, nnue_piece_values_start)
        classical_eval = parse_classical_eval_with_validation(trace_output[classical_values_start+5:classical_values_end])
        final_evaluations = parse_final_evaluations(trace_output[contributions_end+2:])
        return classical_eval, nnue_derived_piece_values, nnue_network_contributions, final_evaluations
    else:
        final_evaluations = parse_final_evaluations(trace_output[contributions_end+1:])
        return nnue_derived_piece_values, nnue_network_contributions, final_evaluations


def parse_classical_eval_with_validation(classical_eval_lines):
    """
    Parses the classical evaluation section of the trace output.

    Args:
        classical_eval_lines (list): Lines of classical evaluation output.

    Returns:
        pd.DataFrame: A DataFrame with MultiIndex columns structured as
                      Phase (MG, EG) and Player (White, Black, Total).
    """
    rows = []
    for line in classical_eval_lines:
        parts = line.strip('|').split('|')
        if len(parts) == 4:
            term, white, black, total = [p.strip() for p in parts]

            # Replace "----" with None and parse values
            def parse_value(value):
                values = value.split()
                return [
                    float(v) if v != "----" else None for v in values
                ] if len(values) == 2 else [None, None]

            white_mg, white_eg = parse_value(white)
            black_mg, black_eg = parse_value(black)
            total_mg, total_eg = parse_value(total)

            # Validate sums (only when values are not None)
            valid_mg = (
                white_mg is not None and black_mg is not None and total_mg is not None and
                abs((white_mg - black_mg) - total_mg) < 0.02
            )
            valid_eg = (
                white_eg is not None and black_eg is not None and total_eg is not None and
                abs((white_eg - black_eg) - total_eg) < 0.02
            )

            # if not (valid_mg and valid_eg):
            #     print(f"Validation failed for {term}: MG or EG totals do not match.")

            rows.append([term, white_mg, black_mg, total_mg, white_eg, black_eg, total_eg])

    # Create MultiIndex columns
    columns = pd.MultiIndex.from_product(
        [["MG", "EG"], ["White", "Black", "Total"]],
        names=["Phase", "Player"]
    )

    # Reformat rows to match column structure
    data = [
        [
            row[1], row[2], row[3],  # MG: White, Black, Total
            row[4], row[5], row[6],  # EG: White, Black, Total
        ]
        for row in rows
    ]

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns, index=[row[0] for row in rows])

    return df


def get_section_indices(trace_output, section_start_marker, section_end_marker=''):
    """
    Automatically determine the start and end indices of a section in trace_output.
    
    Args:
        trace_output (list): List of lines in the trace output.
        section_start_marker (str): The line that marks the start of the section.
        section_end_marker (str): Optional. The line that marks the end of the section. If not provided, the function will include all lines after the start marker until the next blank line or the end of the output.
        
    Returns:
        tuple: (start_index, end_index) of the section in trace_output.
    """
    start_index = -1
    end_index = len(trace_output)  # Default to the end of the list
    
    # Find start index
    for i, line in enumerate(trace_output):
        if section_start_marker in line:
            start_index = i
            break
    
    if start_index == -1:
        for item in trace_output:
            print(item)  # for debugging purposes
        raise ValueError(f"Section start marker '{section_start_marker}' not found in trace output.")
    
    # Find end index if an end marker is provided
    if section_end_marker:
        for i in range(start_index + 1, len(trace_output)):
            if section_end_marker in trace_output[i]:
                end_index = i
                break
    else:
        # End marker not provided: use the next blank line or end of list
        for i in range(start_index + 1, len(trace_output)):
            if not trace_output[i].strip():
                end_index = i
                break
    
    return start_index, end_index


# Function to parse the final evaluations section
def parse_final_evaluations(final_eval_lines):
    """
    Parses the final evaluation section of the trace output.

    Args:
        final_eval_lines (list): Lines containing final evaluation details.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation types and their values.
    """
    rows = []
    for line in final_eval_lines:
        # Use regex to extract the numeric value
        match = re.search(r"(-?\+?\d+\.\d+)", line)
        if match:
            numeric_value = float(match.group(1).replace("+", ""))
            eval_type = line.split()[0] + " " + line.split()[1]  # First two words as type
            rows.append([eval_type, numeric_value])
        else:
            eval_type = " ".join(line.split()[:2])  # First two words as type
            rows.append([eval_type, None])  # No numeric value found

    # Create DataFrame with Evaluation Type as index
    df = pd.DataFrame(rows, columns=["Evaluation Type", "Value"])
    # Check if "Classical_evaluation" exists in the "eval_type" column
    if "Classical evaluation" not in df["Evaluation Type"].values:
        # Create the new row as a DataFrame
        new_row = pd.DataFrame([["Classical evaluation", None]], columns=df.columns)
        
        # Concatenate the new row to the beginning
        df = pd.concat([new_row, df], ignore_index=True)

    return df.set_index("Evaluation Type")


def get_default_options(engine):
    """
    Retrieves all default Stockfish options.

    Args:
        engine: The Stockfish subprocess instance.

    Returns:
        dict: A dictionary of option names and their default values.
    """
    options = {}
    send_command(engine, "uci")
    while True:
        line = engine.stdout.readline().strip()
        if "uciok" in line:
            break
        if line.startswith("option name"):
            parts = line.split(" ")
            name_index = parts.index("name") + 1
            type_index = parts.index("type") + 1
            option_name = " ".join(parts[name_index:parts.index("type")])
            option_type = parts[type_index]

            # Handle hardcoded default values for specific options
            if option_name == "Threads":
                default_value = "7"  # Hardcoded default
            elif option_name == "Hash":
                default_value = "64"  # Hardcoded default
            elif option_name == "UCI_Elo":
                default_value = "3190"  # Hardcoded default
            elif option_name == "SyzygyPath":
                # Construct the path relative to your project
                syzygy_path = os.path.join(
                    os.path.dirname(os.path.abspath("analysis.ipynb")), "syzygy"
                )
                default_value = syzygy_path
            elif option_name == "SyzygyProbeLimit":
                default_value = "5"  # Hardcoded default
            elif "default" in parts:
                # Extract default value if specified in the option
                default_index = parts.index("default") + 1
                default_value = " ".join(parts[default_index:])
            else:
                default_value = None

            # Store the option
            options[option_name] = {
                "type": option_type,
                "default": default_value,
            }

    return options


# Set options as commands
def set_options_as_commands(engine, options):
    """
    Sets Stockfish options using commands.

    Args:
        engine: The Stockfish subprocess instance.
        options (dict): A dictionary of options to set.
    """
    for name, properties in options.items():
        default = properties["default"]
        if default is not None:
            send_command(engine, f'setoption name {name} value {default}')


# Function to send commands to Stockfish
def send_command(engine, command):
    """
    Sends a command to the Stockfish engine.

    Args:
        engine: The Stockfish subprocess instance.
        command: A string command to send to the engine.
    """
    engine.stdin.write(command + "\n")
    engine.stdin.flush()


# Function to continuously read subprocess output
def enqueue_output(out, queue):
    """
    Continuously reads subprocess output and adds it to a queue.

    Args:
        out: The output stream (stdout or stderr) of a subprocess.
        queue: A queue to store the lines of output.
    """
    for line in iter(out.readline, ''):
        queue.put(line.strip())
    out.close()


# Function to read all available output from Stockfish
def read_output_until(pattern, output_queue, timeout=10):
    """
    Reads all available output from a subprocess until a specific pattern is found.

    Args:
        pattern: The target string to stop reading at.
        output_queue: A queue containing subprocess output lines.
        timeout: Maximum time to wait for output (default: 10 seconds).

    Returns:
        list: All lines of output up to and including the target pattern.
    """
    import time
    start_time = time.time()
    output = []
    while True:
        try:
            # Retrieve output from the queue
            line = output_queue.get(timeout=timeout)
            output.append(line)
            # Check if the desired pattern is in the line
            if pattern in line:
                break
        except queue.Empty:
            # If no output is received within the timeout, stop waiting
            if time.time() - start_time > timeout:
                break
    return output


def get_best_move(custom_fen, movetime=1000):
    """
    Runs Stockfish to evaluate a custom FEN position and returns the best move.

    Args:
        exe_path (str): Path to the Stockfish executable.
        custom_fen (str): The FEN string representing the chess position.
        movetime (int, optional): Time in milliseconds for Stockfish to think.

    Returns:
        str: The best move as determined by Stockfish.
    """
    exe_path = STOCKFISH_15_PATH

    # Start Stockfish engine
    engine = subprocess.Popen(
        [exe_path],
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1  # Line-buffered output
    )

    # Get and apply options
    default_options = get_default_options(engine)
    set_options_as_commands(engine, default_options)

    # Queue to hold subprocess output
    output_queue = queue.Queue()

    # Start a thread to read stdout continuously
    thread = threading.Thread(target=enqueue_output, args=(engine.stdout, output_queue))
    thread.daemon = True
    thread.start()

    # Initialize Stockfish in UCI mode
    send_command(engine, "uci")
    read_output_until("uciok", output_queue)  # Wait for "uciok"

    # Set custom position from the provided FEN
    send_command(engine, f"position fen {custom_fen}")

    # Instruct Stockfish to search for the best move for a given movetime
    send_command(engine, f"go movetime {movetime}")

    # Read output until we receive the bestmove line
    bestmove_output = read_output_until("bestmove", output_queue)

    # Extract the best move from the output
    best_move = None
    for line in bestmove_output:
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2:
                best_move = parts[1]
            break

    # Clean up resources
    engine.stdin.close()
    engine.stdout.close()
    engine.terminate()
    thread.join()

    return best_move

if __name__ == "__main__":
    """
    Main entry point for the script. This module defines utility functions for detecting,
    processing, and visualizing chessboards using computer vision.
    """
    print("This is a file containing utility functions for the chess vision project related to Stockfish.")
