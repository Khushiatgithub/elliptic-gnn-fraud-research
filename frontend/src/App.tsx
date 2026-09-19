import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { NavTab, Sidebar } from './components/Sidebar';
import { OverviewPage } from './pages/OverviewPage';
import { DatasetPage } from './pages/DatasetPage';
import { GraphExplorerPage } from './pages/GraphExplorerPage';
import { ModelLabPage } from './pages/ModelLabPage';
import { ExperimentLabPage } from './pages/ExperimentLabPage';
import { PredictionPage } from './pages/PredictionPage';
import { ResearchFindingsPage } from './pages/ResearchFindingsPage';
import { PresentationMode } from './pages/PresentationMode';
import { api } from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>('overview');
  const [isPresentationMode, setIsPresentationMode] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  // Periodic health check
  useEffect(() => {
    const checkBackend = () => {
      api.checkHealth()
        .then(() => setBackendOnline(true))
        .catch(() => setBackendOnline(false));
    };
    checkBackend();
    const interval = setInterval(checkBackend, 15000);
    return () => clearInterval(interval);
  }, []);

  const renderActivePage = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPage />;
      case 'dataset':
        return <DatasetPage />;
      case 'graph':
        return <GraphExplorerPage />;
      case 'models':
        return <ModelLabPage />;
      case 'experiments':
        return <ExperimentLabPage />;
      case 'prediction':
        return <PredictionPage />;
      case 'findings':
        return <ResearchFindingsPage />;
      default:
        return <OverviewPage />;
    }
  };

  return (
    <div className="min-h-screen bg-lab-bg text-lab-text flex flex-col font-sans">
      {/* Top Navigation */}
      <Navbar
        isPresentationMode={isPresentationMode}
        onTogglePresentation={() => setIsPresentationMode((prev) => !prev)}
        backendOnline={backendOnline}
      />

      {/* Main Content Area with Sidebar */}
      <div className="flex-1 flex overflow-hidden">
        <Sidebar activeTab={activeTab} onSelectTab={setActiveTab} />

        <main className="flex-1 overflow-y-auto bg-lab-bg">
          {renderActivePage()}
        </main>
      </div>

      {/* Full-Screen Presentation Mode Overlay */}
      {isPresentationMode && (
        <PresentationMode onExit={() => setIsPresentationMode(false)} />
      )}
    </div>
  );
};

export default App;
