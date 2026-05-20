import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from coperception.models.det.LSTMAutoencoder import LSTMAE

DEFAULT_DATA_PATH = "/data2/user2/yihang/GCP/coperception/logs/scene_0_ego_1.npz"
DEFAULT_SAVE_PATH = "/data2/user2/yihang/GCP/coperception/logs/model/"


class BEVFlowDataset(Dataset):
    def __init__(self, data_path, seq_length=5, iou_threshold=0.3, max_objects=10):
        self.seq_length = seq_length
        self.iou_threshold = iou_threshold
        self.max_objects = max_objects
        self.sequences = []
        self.load_data(data_path)
        
    def match_consecutive_frames(self, det1, det2):
        """
        Match detection boxes between two consecutive frames
        Args:
            det1: First frame detection boxes (N, 9) - contains 8 coordinate values and 1 confidence
            det2: Second frame detection boxes (M, 9)
        Returns:
            matched_det1, matched_det2: Matched detection boxes
        """
        N, M = len(det1), len(det2)
        if N == 0 or M == 0:
            return np.array([]), np.array([])
            
        # Calculate IoU matrix
        iou_matrix = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                # Use 8 corner coordinates to calculate IoU
                corners1 = det1[i,:8].reshape(4, 2)  # Convert to 4 corner coordinates
                corners2 = det2[j,:8].reshape(4, 2)
                iou_matrix[i,j] = self.calculate_iou_from_corners(corners1, corners2)
        
        # Use Hungarian algorithm to match
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        # Only keep matches with IoU greater than threshold
        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r,c] >= self.iou_threshold:
                matched_pairs.append((r,c))
                
        if not matched_pairs:
            return np.array([]), np.array([])
            
        matched_idx1, matched_idx2 = zip(*matched_pairs)
        return det1[list(matched_idx1)], det2[list(matched_idx2)]
    
    def calculate_iou_from_corners(self, corners1, corners2):
        """
        Calculate IoU using corner coordinates
        Args:
            corners1: (4, 2) First box's four corner coordinates
            corners2: (4, 2) Second box's four corner coordinates
        Returns:
            iou: IoU value
        """
        # Convert corner coordinates to polygon
        from shapely.geometry import Polygon
        
        poly1 = Polygon(corners1)
        poly2 = Polygon(corners2)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
            
        # Calculate intersection and union
        intersection = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - intersection
        
        if union == 0:
            return 0.0
            
        return intersection / union
        
    def match_sequence(self, frames):
        """
        Match consecutive seq_length frames
        Args:
            frames: list of arrays, each array contains a frame's detection boxes
        Returns:
            matched_sequence: Matched sequence
        """
        if len(frames) < self.seq_length:
            return None
            
        # Match from the first frame
        current_boxes = frames[0]
        matched_sequence = [current_boxes]
        
        # Match frame by frame
        for i in range(1, len(frames)):
            matched_curr, matched_next = self.match_consecutive_frames(
                current_boxes, frames[i])
            
            if len(matched_curr) == 0:
                return None
                
            matched_sequence.append(matched_next)
            current_boxes = matched_next
            
        # Ensure all frames have the same number of detection boxes
        if not all(len(frame) == len(matched_sequence[0]) for frame in matched_sequence):
            return None
            
        return matched_sequence
    def load_data(self, data_path):
        """Load and process data"""
        try:
            data = np.load(data_path, allow_pickle=True)
            bev_flows = data['bev_flows']
            
            # Create sliding window sequence and match
            for i in range(len(bev_flows) - self.seq_length + 1):
                sequence = bev_flows[i:i + self.seq_length]
                matched_sequence = self.match_sequence(sequence)
                
                if matched_sequence is not None and len(matched_sequence) == self.seq_length:
                    # Create sequence for each detection box
                    for obj_idx in range(len(matched_sequence[0])):
                        # Extract trajectory of a single object
                        single_obj_sequence = []
                        for frame in matched_sequence:
                            single_obj_sequence.append(frame[obj_idx:obj_idx+1])  # Keep 2D shape
                        
                        # Convert to tensor and add to sequence list
                        single_obj_sequence = torch.FloatTensor(np.array(single_obj_sequence))
                        self.sequences.append(single_obj_sequence)
            
            print(f"Loaded {len(self.sequences)} valid sequences from {data_path}")
            if len(self.sequences) > 0:
                print(f"Single sequence shape: {self.sequences[0].shape}")  # Should be [seq_length, 1, features]
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx]

def train_lstm_ae(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 8
    hidden_dim = 32
    num_layers = 2

    dataset = BEVFlowDataset(data_path=args.data_path, seq_length=args.seq_length)
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
        
    print(f"Dataset size: {len(dataset)}")
    print(f"Single sequence shape: {dataset[0].shape}")
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    model = LSTMAE(input_dim=input_dim, 
                   hidden_dim=hidden_dim, 
                   num_layers=num_layers, 
                   seq_length=args.seq_length).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for sequences in train_loader:
            sequences = sequences.squeeze(2)
            sequences = sequences.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(sequences)
            loss = F.mse_loss(reconstructed, sequences)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validate
        val_loss = validate_lstm_ae(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            print(f"Save best model to {model_path}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Save checkpoint to {checkpoint_path}")
        
        print(f"Epoch {epoch}: Train Loss = {total_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")

def validate_lstm_ae(model, val_loader, device):
    """Validation function"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for sequences in val_loader:
            batch_size = sequences.size(0)
            seq_length = sequences.size(1)
            num_objects = sequences.size(2)
            features = sequences.size(3)
            
            sequences_reshaped = sequences.transpose(1, 2).contiguous()
            sequences_reshaped = sequences_reshaped.view(batch_size * num_objects, seq_length, features).to(device)
            reconstructed = model(sequences_reshaped)
            
            reconstructed = reconstructed.view(batch_size, num_objects, seq_length, features)
            sequences = sequences.to(device)
            
            loss = F.mse_loss(reconstructed, sequences.transpose(1, 2))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM-AE for GCP temporal defense")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="BEV flow data path")
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH, help="Model save path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=5, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()

if __name__ == "__main__":
    train_lstm_ae(parse_args())
