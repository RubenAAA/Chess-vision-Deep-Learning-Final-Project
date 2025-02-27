from ultralytics import YOLO
import torch


def train_board_detection():
    """
    Trains a YOLOv8 model for detecting chessboard intersections and saves the trained model.

    Steps:
    1. Load a pretrained YOLOv8 model.
    2. Train the model using the dataset specified in 'board_config.yml'.
    3. Configure training parameters such as epochs, batch size, image size, and device settings.
    4. Validate the trained model to assess its performance.
    5. Save the best model weights for future inference.
    6. Export the trained model in TorchScript format for deployment.
    7. Print validation metrics for performance evaluation.
    """

    # Step 1: Load a pretrained YOLOv8 model
    model_board = YOLO("yolo11n.pt")  

    # Step 2: Train the model using the specified dataset and parameters
    model_board.train(
        data="./board_config.yml",  # Path to dataset configuration file
        epochs=50,                # Number of training epochs
        batch=60,                 # Batch size
        imgsz=480,                # Image resolution
        device=0,                 # Use GPU (device 0)
        pretrained=True,          # Use pretrained weights for transfer learning
        optimizer="auto",         # Automatically select the best optimizer
        seed=0,                   # Set random seed for reproducibility
        deterministic=True,       # Ensure deterministic behavior for consistency
        workers=8,                # Number of parallel workers for data loading
        project="runs/detect_board",  # Project directory for saving results
        name="train"              # Name of the training run
    )

    # Step 3: Validate the trained model and save performance metrics
    metrics = model_board.val(project="runs/detect_board", name="val")  # Run validation

    # Step 4: Save the best trained model for future use
    model_board.save("runs/detect_board/train/weights/best.pt")

    # Step 5: Export the trained model in TorchScript format for deployment
    model_board.export(format="torchscript", project="runs/detect_board", name="export")  

    # Step 6: Print the validation performance metrics
    print(metrics)


def train_piece_detection():
    """
    Trains a YOLOv8 model for detecting chess pieces and saves the trained model.

    Steps:
    1. Load a pretrained YOLOv8 model.
    2. Train the model using the dataset specified in 'pieces_config.yml'.
    3. Configure training parameters such as epochs, batch size, image size, and device settings.
    4. Validate the trained model to assess its performance.
    5. Save the best model weights for future inference.
    6. Export the trained model in TorchScript format for deployment.
    7. Print validation metrics for performance evaluation.
    """

    # Step 1: Load a pretrained YOLOv8 model
    model_pieces = YOLO("yolo11n.pt")  

    # Step 2: Train the model using the specified dataset and parameters
    model_pieces.train(
        data="./pieces_config.yml",  # Path to dataset configuration file
        epochs=50,                # Number of training epochs
        batch=60,                 # Batch size
        imgsz=480,                # Image resolution
        device=0,                 # Use GPU (device 0)
        pretrained=True,          # Use pretrained weights for transfer learning
        optimizer="auto",         # Automatically select the best optimizer
        seed=0,                   # Set random seed for reproducibility
        deterministic=True,       # Ensure deterministic behavior for consistency
        workers=8,                # Number of parallel workers for data loading
        project="runs/detect_pieces",  # Project directory for saving results
        name="train"              # Name of the training run
    )

    # Step 3: Validate the trained model and save performance metrics
    metrics = model_pieces.val(project="runs/detect_pieces", name="val")  # Run validation

    # Step 4: Save the best trained model for future use
    model_pieces.save("runs/detect_pieces/train/weights/best.pt")

    # Step 5: Export the trained model in TorchScript format for deployment
    model_pieces.export(format="torchscript", project="runs/detect_pieces", name="export")  

    # Step 6: Print the validation performance metrics
    print(metrics)


if __name__ == "__main__":
    print("Starting Training:")
    train_board_detection()
    train_piece_detection()
    print("Finished Training")
    