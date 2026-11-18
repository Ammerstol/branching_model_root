import numpy as np
from collections import Counter

def run_branching_simulation(R, k, nu, s, max_tree_size=10000):
    """
    Simulates a branching process that stops only after a full 
    discrete generation is completed.
    
    Args:
        max_tree_size: The soft target. We stop after the generation 
                       where the total size exceeds this value.
    """
    # Negative Binomial parameters
    p_param = k / (k + R)
    n_param = k
    
    # State tracking
    # current_gen_ids: The list of genotype IDs alive in the current generation
    current_gen_ids = [0] 
    
    # all_recorded_ids: Master list of everyone ever born
    all_recorded_ids = [0]
    next_genotype_id = 1
    
    # --- Generational Loop ---
    # We continue as long as the total size is below target AND the epidemic isn't dead
    while len(all_recorded_ids) < max_tree_size and len(current_gen_ids) > 0:
        
        next_gen_ids = []
        
        # Process EVERY individual in the current generation
        for parent_id in current_gen_ids:
            
            # Draw offspring count for this parent
            num_offspring = np.random.negative_binomial(n_param, p_param)
            
            if num_offspring > 0:
                # Create offspring with Allele Mutation Model
                for _ in range(num_offspring):
                    if np.random.rand() < nu:
                        child_id = next_genotype_id
                        next_genotype_id += 1
                    else:
                        child_id = parent_id
                    
                    next_gen_ids.append(child_id)
                    all_recorded_ids.append(child_id)
        
        # Advance to the next generation
        current_gen_ids = next_gen_ids

    # --- Bernoulli Thinning ---
    total_nodes = len(all_recorded_ids)
    
    # Apply sampling fraction 's' to the full biological tree
    mask = np.random.rand(total_nodes) < s
    sampled_genotypes = np.array(all_recorded_ids)[mask]
    
    if len(sampled_genotypes) == 0:
        return [], total_nodes
        
    counts = Counter(sampled_genotypes)
    return list(counts.values()), total_nodes