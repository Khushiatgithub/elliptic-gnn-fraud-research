import axios from 'axios';
import {
  DatasetSummary,
  TimestepInfo,
  TransactionDetail,
  SampleTransaction,
  SubgraphResponse,
  MainResultRecord,
  FeatureComparisonRecord,
  WeightingComparisonRecord,
  AblationRecord,
  SeedStabilityRecord,
  StatisticalTestRecord,
  ErrorAnalysisRecord,
  ExperimentInventoryRecord,
  ModelInfo,
  PredictionResponse,
  FigureInfo
} from '../types/api';

const API_BASE = 'http://127.0.0.1:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Health
  checkHealth: async () => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  // Dataset
  getSummary: async (): Promise<DatasetSummary> => {
    const res = await apiClient.get<DatasetSummary>('/dataset/summary');
    return res.data;
  },

  getTimesteps: async (): Promise<TimestepInfo[]> => {
    const res = await apiClient.get<TimestepInfo[]>('/dataset/timesteps');
    return res.data;
  },

  getSampleTransactions: async (): Promise<SampleTransaction[]> => {
    const res = await apiClient.get<SampleTransaction[]>('/dataset/sample-transactions');
    return res.data;
  },

  getTransaction: async (txId: number): Promise<TransactionDetail> => {
    const res = await apiClient.get<TransactionDetail>(`/dataset/transaction/${txId}`);
    return res.data;
  },

  // Graph
  getNeighborhood: async (txId: number, kHop: number = 1, maxNodes: number = 80): Promise<SubgraphResponse> => {
    const res = await apiClient.get<SubgraphResponse>(`/transaction/${txId}/neighbors`, {
      params: { k_hop: kHop, max_nodes: maxNodes }
    });
    return res.data;
  },

  // Results
  getMainResults: async (): Promise<MainResultRecord[]> => {
    const res = await apiClient.get<MainResultRecord[]>('/results/main');
    return res.data;
  },

  getFeatureComparison: async (): Promise<FeatureComparisonRecord[]> => {
    const res = await apiClient.get<FeatureComparisonRecord[]>('/results/features');
    return res.data;
  },

  getWeightingComparison: async (): Promise<WeightingComparisonRecord[]> => {
    const res = await apiClient.get<WeightingComparisonRecord[]>('/results/weighting');
    return res.data;
  },

  getAblations: async (ablationType?: string): Promise<AblationRecord[]> => {
    const res = await apiClient.get<AblationRecord[]>('/results/ablations', {
      params: ablationType ? { ablation_type: ablationType } : {}
    });
    return res.data;
  },

  getSeedStability: async (modelFilter?: string): Promise<SeedStabilityRecord[]> => {
    const res = await apiClient.get<SeedStabilityRecord[]>('/results/seeds', {
      params: modelFilter ? { model: modelFilter } : {}
    });
    return res.data;
  },

  getStatisticalTests: async (): Promise<StatisticalTestRecord[]> => {
    const res = await apiClient.get<StatisticalTestRecord[]>('/results/statistics');
    return res.data;
  },

  getErrorAnalysis: async (): Promise<ErrorAnalysisRecord[]> => {
    const res = await apiClient.get<ErrorAnalysisRecord[]>('/results/error-analysis');
    return res.data;
  },

  getInventory: async (): Promise<ExperimentInventoryRecord[]> => {
    const res = await apiClient.get<ExperimentInventoryRecord[]>('/results/inventory');
    return res.data;
  },

  // Prediction & Models
  getAvailableModels: async (): Promise<ModelInfo[]> => {
    const res = await apiClient.get<ModelInfo[]>('/models/available');
    return res.data;
  },

  predict: async (txId: number, modelId: string): Promise<PredictionResponse> => {
    const res = await apiClient.post<PredictionResponse>('/predict', {
      tx_id: txId,
      model_id: modelId
    });
    return res.data;
  },

  // Figures
  getFiguresList: async (): Promise<FigureInfo[]> => {
    const res = await apiClient.get<FigureInfo[]>('/figures/list');
    return res.data;
  },

  getFigureUrl: (filename: string): string => {
    return `${API_BASE}/figures/${filename}`;
  }
};
