"""
Benjamini-Hochberg Decision Module for GCP

This module provides BH-based decision making as an alternative/complement
to the threshold-based approach in GCP.

The BH procedure controls False Discovery Rate (FDR) when testing multiple
hypotheses (i.e., whether each agent is malicious).
"""

import os

import numpy as np


class BHDecisionModule:
    """
    BH Decision Module for malicious agent detection.
    
    This module maintains a calibration set of normal agent scores and uses
    conformal p-values + BH procedure to decide which agents to reject.
    """
    
    def __init__(self, alpha_bh=0.2, calibration_path=None, allow_default_calibration=False):
        """
        Initialize BH Decision Module.
        
        Args:
            alpha_bh: FDR control level (default 0.2)
            calibration_path: Path to calibration set file.
            allow_default_calibration: If True, build a synthetic default calibration
                when no valid calibration file is provided.
        """
        self.alpha_bh = alpha_bh
        self.calibration_scores = None
        self.calibration_path = calibration_path
        self.calibration_source = "none"

        if calibration_path and os.path.exists(calibration_path):
            self.load_calibration(calibration_path)
        elif allow_default_calibration:
            self._build_default_calibration()
    
    def _build_default_calibration(self):
        """
        Build default calibration set to approximate GCP threshold behavior.
        
        Based on analysis:
        - GCP threshold: total_loss > 0.38 (fail)
        - Benign samples typically have scores in range [0, 0.4]
        - We construct calibration set with max_score ≈ 0.38 to match threshold
        """
        # Calibration set represents "normal" agent behavior scores
        # Constructed based on empirical analysis of benign agent scores
        np.random.seed(42)
        
        # Parameters tuned to match GCP threshold behavior
        # Mean around 0.15-0.20, with max around 0.38 (the threshold)
        n_samples = 200
        
        # Mixture of low scores (typical benign) and some higher ones (edge cases)
        low_scores = np.random.beta(2, 8, size=int(n_samples * 0.7)) * 0.3  # 0-0.3
        mid_scores = np.random.beta(3, 3, size=int(n_samples * 0.3)) * 0.15 + 0.20  # 0.2-0.35
        
        self.calibration_scores = np.concatenate([low_scores, mid_scores])
        self.calibration_scores = np.clip(self.calibration_scores, 0, 0.38)
        self.calibration_source = "synthetic_default"
        
        # Stats
        self._calibration_mean = np.mean(self.calibration_scores)
        self._calibration_max = np.max(self.calibration_scores)
        
    def load_calibration(self, path):
        """Load calibration set from file."""
        data = np.load(path)
        self.calibration_scores = data['scores']
        self._calibration_mean = np.mean(self.calibration_scores)
        self._calibration_max = np.max(self.calibration_scores)
        self.calibration_source = path
        
    def save_calibration(self, path):
        """Save calibration set to file."""
        np.savez(path, scores=self.calibration_scores)
        self.calibration_source = path

    def has_calibration(self):
        """Return whether a valid calibration set is available."""
        return self.calibration_scores is not None and len(self.calibration_scores) > 0
        
    def update_calibration(self, scores, labels=None):
        """
        Update calibration set with new benign scores.
        
        Args:
            scores: Array of scores from agents
            labels: Optional boolean array, True=attack. Only benign scores are used.
        """
        if labels is not None:
            benign_scores = scores[~labels]
        else:
            benign_scores = scores
            
        if self.calibration_scores is None:
            self.calibration_scores = benign_scores
        else:
            # Keep recent history with exponential decay
            decay = 0.95
            old_weight = decay * len(self.calibration_scores)
            new_weight = len(benign_scores)
            total = old_weight + new_weight
            
            # Subsample to keep calibration set manageable
            max_size = 500
            if len(self.calibration_scores) + len(benign_scores) > max_size:
                n_keep = int(max_size * old_weight / total)
                idx = np.random.choice(len(self.calibration_scores), n_keep, replace=False)
                self.calibration_scores = np.concatenate([
                    self.calibration_scores[idx], 
                    benign_scores
                ])
            else:
                self.calibration_scores = np.concatenate([
                    self.calibration_scores, 
                    benign_scores
                ])
                
        self._calibration_mean = np.mean(self.calibration_scores)
        self._calibration_max = np.max(self.calibration_scores)
    
    def compute_pvalue(self, score):
        """
        Compute conformal p-value for a single score.
        
        P-value = proportion of calibration scores >= test score
        Higher score means more anomalous, so smaller p-value.
        """
        if not self.has_calibration():
            return 0.5  # Default if no calibration
            
        return (1 + np.sum(self.calibration_scores >= score)) / (1 + len(self.calibration_scores))
    
    def benjamini_hochberg(self, p_values):
        """
        Apply Benjamini-Hochberg procedure to control FDR.
        
        Args:
            p_values: Array of p-values for each agent
            
        Returns:
            rejected: Boolean array, True if agent should be rejected
        """
        m = len(p_values)
        if m == 0:
            return np.array([], dtype=bool)
            
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Find largest k where p_(k) <= k/m * alpha
        threshold_idx = -1
        for k in range(m):
            if sorted_p[k] <= (k + 1) / m * self.alpha_bh:
                threshold_idx = k
        
        rejected = np.zeros(m, dtype=bool)
        if threshold_idx >= 0:
            rejected[sorted_indices[:threshold_idx + 1]] = True
            
        return rejected
    
    def decide(self, agent_scores, agent_indices=None):
        """
        Make rejection decision for a set of agents using BH procedure.
        
        Args:
            agent_scores: Dict or array of {agent_idx: score} or [score1, score2, ...]
            agent_indices: Optional list of agent indices (if scores is array)
            
        Returns:
            rejected_agents: List of agent indices that should be rejected
            decisions: Dict of {agent_idx: {'rejected': bool, 'score': float, 'pvalue': float}}
        """
        if isinstance(agent_scores, dict):
            indices = list(agent_scores.keys())
            scores = np.array([agent_scores[i] for i in indices])
        else:
            scores = np.array(agent_scores)
            indices = agent_indices if agent_indices else list(range(len(scores)))
        
        if len(scores) == 0:
            return [], {}
            
        if not self.has_calibration():
            decisions = {
                indices[i]: {
                    'rejected': False,
                    'score': scores[i],
                    'pvalue': None,
                    'method': 'threshold_required'
                }
                for i in range(len(indices))
            }
            return [], decisions

        # Compute p-values
        p_values = np.array([self.compute_pvalue(s) for s in scores])
        
        # Apply BH procedure
        rejected = self.benjamini_hochberg(p_values)
        
        # Build results
        rejected_agents = [indices[i] for i in range(len(indices)) if rejected[i]]
        decisions = {
            indices[i]: {
                'rejected': rejected[i],
                'score': scores[i],
                'pvalue': p_values[i]
            }
            for i in range(len(indices))
        }
        
        return rejected_agents, decisions
    
    def decide_single(self, score, threshold_fallback=0.38):
        """
        Single agent decision (when only one agent to check).
        
        For single agent, BH reduces to simple p-value threshold.
        We use p-value < alpha as rejection criterion.
        
        Args:
            score: The agent's anomaly score
            threshold_fallback: Fallback threshold if calibration unavailable
            
        Returns:
            rejected: Boolean
            info: Dict with score, pvalue, etc.
        """
        if not self.has_calibration():
            # Fallback to threshold
            rejected = score > threshold_fallback
            return rejected, {
                'score': score,
                'pvalue': None,
                'method': 'threshold',
                'threshold': threshold_fallback
            }
            
        p_value = self.compute_pvalue(score)
        rejected = p_value <= self.alpha_bh
        
        return rejected, {'score': score, 'pvalue': p_value, 'method': 'bh'}
    
    def get_stats(self):
        """Get calibration set statistics."""
        if not self.has_calibration():
            return {
                'size': 0,
                'mean': None,
                'max': None,
                'alpha_bh': self.alpha_bh,
                'source': self.calibration_source
            }
        return {
            'size': len(self.calibration_scores),
            'mean': self._calibration_mean,
            'max': self._calibration_max,
            'alpha_bh': self.alpha_bh,
            'source': self.calibration_source
        }


def create_calibration_set_from_logs(log_dir, output_path, threshold=0.38):
    """
    Create calibration set from experiment logs.
    
    Uses benign samples (non-attack frames) with scores below threshold.
    """
    import glob
    import re
    
    all_benign_scores = []
    
    for log_file in glob.glob(os.path.join(log_dir, '*0attackers*.txt')):
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract spatio-temporal detection loss values
        losses = re.findall(r'Total spatio-temporal detection loss: ([\d.]+)', content)
        for loss in losses:
            score = float(loss)
            if score < threshold:  # Only benign-range scores
                all_benign_scores.append(score)
    
    if len(all_benign_scores) > 0:
        scores = np.array(all_benign_scores)
        np.savez(output_path, scores=scores)
        print(f"Created calibration set with {len(scores)} samples")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Max: {np.max(scores):.4f}")
        return scores
    else:
        print("No benign samples found in logs")
        return None


if __name__ == '__main__':
    # Test the module
    bh = BHDecisionModule(alpha_bh=0.2)
    print("Calibration stats:", bh.get_stats())
    
    # Test with some scores
    test_scores = {
        0: 0.65,  # High score (likely attack)
        2: 0.58,  # High score (likely attack)
        3: 0.15,  # Low score (benign)
        4: 0.10,  # Low score (benign)
        5: 0.42,  # Medium score (borderline)
    }
    
    rejected, decisions = bh.decide(test_scores)
    print(f"\nTest scores: {test_scores}")
    print(f"Rejected agents: {rejected}")
    for agent, info in decisions.items():
        print(f"  Agent {agent}: rejected={info['rejected']}, score={info['score']:.3f}, pvalue={info['pvalue']:.4f}")
