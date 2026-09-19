export interface DatasetSummary {
  total_transactions: number;
  total_edges: number;
  total_timesteps: number;
  total_features: number;
  local_features: number;
  aggregated_features: number;
  illicit_count: number;
  licit_count: number;
  unknown_count: number;
  labeled_count: number;
  illicit_labeled_ratio: number;
  train_timesteps: string;
  val_timesteps: string;
  test_timesteps: string;
  train_count: number;
  train_illicit_count: number;
  val_count: number;
  val_illicit_count: number;
  test_count: number;
  test_illicit_count: number;
}

export interface TimestepInfo {
  timestep: number;
  total_txs: number;
  licit_txs: number;
  illicit_txs: number;
  unknown_txs: number;
  labeled_txs: number;
  illicit_ratio: number;
  split: 'train' | 'val' | 'test';
}

export interface TransactionDetail {
  tx_id: number;
  timestep: number;
  split: string;
  ground_truth_label: 'illicit' | 'licit' | 'unknown';
  in_degree: number;
  out_degree: number;
  total_degree: number;
  features_local: Record<string, number>;
  features_agg: Record<string, number>;
}

export interface SampleTransaction {
  tx_id: number;
  timestep: number;
  split: string;
  label: string;
  description: string;
}

export interface GraphNode {
  id: string;
  tx_id: number;
  label: 'illicit' | 'licit' | 'unknown';
  timestep: number;
  split: string;
  in_degree: number;
  out_degree: number;
  total_degree: number;
  is_center: boolean;
  hop: number;
  x?: number;
  y?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  source_tx_id: number;
  target_tx_id: number;
}

export interface SubgraphResponse {
  center_tx_id: number;
  k_hop: number;
  total_nodes: number;
  total_edges: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  center_node?: GraphNode;
}

export interface MainResultRecord {
  model: string;
  family: string;
  features: string;
  weighting: string;
  val_f1_mean: number;
  val_f1_std: number;
  val_prauc_mean: number;
  val_prauc_std: number;
  test_f1_mean: number;
  test_f1_std: number;
  test_prauc_mean: number;
  test_prauc_std: number;
  test_precision_mean: number;
  test_precision_std: number;
  test_recall_mean: number;
  test_recall_std: number;
  test_mcc_mean: number;
  test_mcc_std: number;
  seeds: string;
}

export interface FeatureComparisonRecord {
  model: string;
  feature_space: string;
  feature_count: number;
  val_f1: string;
  val_prauc: string;
  test_f1: string;
  test_prauc: string;
  test_mcc: string;
  delta_f1: string;
  delta_prauc: string;
  delta_mcc: string;
}

export interface WeightingComparisonRecord {
  model: string;
  feature_space: string;
  weighting: string;
  val_f1: string;
  val_prauc: string;
  test_f1: string;
  test_prauc: string;
  test_precision: string;
  test_recall: string;
  test_mcc: string;
  delta_f1: string;
  delta_prauc: string;
  delta_precision: string;
  delta_recall: string;
  delta_mcc: string;
}

export interface AblationRecord {
  ablation: string;
  model: string;
  setting: string;
  test_f1_formatted: string;
  test_prauc_formatted: string;
  test_precision_formatted: string;
  test_recall_formatted: string;
  test_mcc_formatted: string;
  test_f1_mean: number;
  test_f1_std: number;
  test_prauc_mean: number;
  test_prauc_std: number;
  interpretation: string;
}

export interface SeedStabilityRecord {
  model: string;
  phase: string;
  seed: number;
  val_f1: number;
  val_prauc: number;
  test_f1: number;
  test_prauc: number;
  test_mcc: number;
  test_f1_mean: number;
  test_f1_std: number;
  test_f1_min: number;
  test_f1_max: number;
  test_f1_range: number;
  test_prauc_mean: number;
  test_prauc_std: number;
  test_prauc_range: number;
}

export interface StatisticalTestRecord {
  model_a: string;
  model_b: string;
  metric: string;
  gnn_mean_std: string;
  baseline_mean_std: string;
  mean_difference: number;
  paired_t_statistic: number;
  paired_t_test_p_raw: number;
  holm_adjusted_p: number;
  wilcoxon_w_statistic: number;
  wilcoxon_p_value: number;
  paired_cohen_dz: number;
  sample_size_n: number;
  notes: string;
}

export interface ErrorAnalysisRecord {
  domain: string;
  category_or_timestep: string;
  sample_count: string;
  proportion_or_rate: string;
  mean_in_degree: string;
  mean_out_degree: string;
  precision: string;
  recall: string;
  illicit_f1: string;
  key_observation: string;
}

export interface ExperimentInventoryRecord {
  experiment_id: string;
  phase: string;
  model_family: string;
  model: string;
  feature_space: string;
  feature_count: number;
  weighting: string;
  graph_context: string;
  graph_direction: string;
  depth: string;
  seed: number;
  train_timesteps: string;
  val_timesteps: string;
  test_timesteps: string;
  val_f1: number;
  val_prauc: number;
  test_f1: number;
  test_prauc: number;
  test_precision: number;
  test_recall: number;
  test_mcc: number;
  status: string;
  section: string;
  notes: string;
}

export interface ModelInfo {
  model_id: string;
  name: string;
  family: string;
  is_gnn: boolean;
  feature_mode: string;
  optimal_threshold: number;
  description: string;
  available: boolean;
}

export interface FeatureContribution {
  feature_name: string;
  value: number;
  importance_or_weight: number;
  description: string;
}

export interface PredictionRequest {
  tx_id: number;
  model_id: string;
}

export interface PredictionResponse {
  tx_id: number;
  timestep: number;
  split: string;
  ground_truth_label: string;
  model_id: string;
  model_name: string;
  model_family: string;
  is_gnn: boolean;
  predicted_probability: number;
  classification_threshold: number;
  predicted_class: number;
  predicted_label: string;
  confidence_score: number;
  in_degree: number;
  out_degree: number;
  neighborhood_size_1hop: number;
  feature_contributions?: FeatureContribution[];
  graph_context?: {
    subgraph_nodes_2hop: number;
    subgraph_edges_2hop: number;
    direct_1hop_neighbors: number;
    labeled_neighbors_count: number;
    unknown_neighbors_count: number;
    neighbor_class_composition: {
      illicit: number;
      licit: number;
      unknown: number;
    };
    homophily_ratio: number | string;
  };
}

export interface FigureInfo {
  filename: string;
  title: string;
  category: string;
  description: string;
  paper_section: string;
  url: string;
}
