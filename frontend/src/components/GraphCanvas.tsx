import React, { useState, useEffect, useRef, useMemo } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Info, Maximize2 } from 'lucide-react';
import { GraphEdge, GraphNode, SubgraphResponse } from '../types/api';

interface GraphCanvasProps {
  subgraph: SubgraphResponse | null;
  selectedNode: GraphNode | null;
  onSelectNode: (node: GraphNode) => void;
  loading?: boolean;
}

interface Point {
  x: number;
  y: number;
}

export const GraphCanvas: React.FC<GraphCanvasProps> = ({
  subgraph,
  selectedNode,
  onSelectNode,
  loading = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<Point>({ x: 0, y: 0 });
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);

  // Reset zoom & pan when center transaction changes
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [subgraph?.center_tx_id]);

  // Compute layout positions for nodes in a clean radial force layout
  const layoutNodes = useMemo(() => {
    if (!subgraph || subgraph.nodes.length === 0) return [];

    const width = 800;
    const height = 600;
    const centerX = width / 2;
    const centerY = height / 2;

    const centerNode = subgraph.nodes.find((n) => n.is_center);
    const hop1Nodes = subgraph.nodes.filter((n) => !n.is_center && n.hop === 1);
    const hop2Nodes = subgraph.nodes.filter((n) => !n.is_center && n.hop === 2);

    const positions: Record<string, Point> = {};

    if (centerNode) {
      positions[centerNode.id] = { x: centerX, y: centerY };
    }

    // Distribute 1-hop nodes around inner circle
    const r1 = 160;
    hop1Nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / Math.max(1, hop1Nodes.length);
      positions[node.id] = {
        x: centerX + r1 * Math.cos(angle),
        y: centerY + r1 * Math.sin(angle),
      };
    });

    // Distribute 2-hop nodes around outer circle
    const r2 = 270;
    hop2Nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / Math.max(1, hop2Nodes.length) + 0.15;
      positions[node.id] = {
        x: centerX + r2 * Math.cos(angle),
        y: centerY + r2 * Math.sin(angle),
      };
    });

    return subgraph.nodes.map((node) => ({
      ...node,
      x: positions[node.id]?.x ?? centerX,
      y: positions[node.id]?.y ?? centerY,
    }));
  }, [subgraph]);

  // Pan handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left click
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom((prev) => Math.min(Math.max(0.4, prev * zoomFactor), 3.0));
  };

  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const getNodeColor = (label: string, isCenter: boolean) => {
    if (isCenter) return '#6366f1'; // Indigo center
    if (label === 'illicit') return '#f43f5e'; // Rose illicit
    if (label === 'licit') return '#10b981'; // Emerald licit
    return '#64748b'; // Slate unknown
  };

  const getNodeGlow = (label: string, isCenter: boolean) => {
    if (isCenter) return 'rgba(99, 102, 241, 0.4)';
    if (label === 'illicit') return 'rgba(244, 63, 94, 0.35)';
    if (label === 'licit') return 'rgba(16, 185, 129, 0.35)';
    return 'rgba(100, 116, 139, 0.2)';
  };

  return (
    <div className="relative w-full h-full bg-[#07090e] rounded-xl border border-lab-border overflow-hidden select-none">
      {/* Background Grid Pattern */}
      <div
        className="absolute inset-0 opacity-15 pointer-events-none"
        style={{
          backgroundImage:
            'radial-gradient(#2a3754 1px, transparent 1px), radial-gradient(#2a3754 1px, #07090e 1px)',
          backgroundSize: '24px 24px',
          backgroundPosition: '0 0, 12px 12px',
        }}
      />

      {/* Control Buttons */}
      <div className="absolute top-4 right-4 z-20 flex flex-col space-y-1.5 bg-lab-surface/90 border border-lab-border p-1.5 rounded-lg shadow-xl backdrop-blur-md">
        <button
          onClick={() => setZoom((z) => Math.min(z * 1.2, 3.0))}
          className="p-1.5 text-slate-300 hover:text-white hover:bg-lab-card rounded transition-colors"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(z * 0.8, 0.4))}
          className="p-1.5 text-slate-300 hover:text-white hover:bg-lab-card rounded transition-colors"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={handleResetView}
          className="p-1.5 text-slate-300 hover:text-white hover:bg-lab-card rounded transition-colors"
          title="Reset View"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Legend Badge */}
      <div className="absolute bottom-4 left-4 z-20 flex items-center space-x-4 bg-lab-surface/90 border border-lab-border px-3 py-2 rounded-lg text-xs font-mono backdrop-blur-md">
        <div className="flex items-center space-x-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]" />
          <span className="text-slate-300">Illicit</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
          <span className="text-slate-300">Licit</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
          <span className="text-slate-400">Unknown</span>
        </div>
        <div className="flex items-center space-x-1.5 border-l border-lab-border pl-3">
          <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 ring-2 ring-indigo-400/50" />
          <span className="text-indigo-300 font-semibold">Center Node</span>
        </div>
      </div>

      {/* Interactive SVG Canvas */}
      <div
        ref={containerRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <svg className="w-full h-full">
          <defs>
            {/* Arrow Marker Definition */}
            <marker
              id="arrowhead"
              markerWidth="8"
              markerHeight="6"
              refX="14"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 8 3, 0 6" fill="#3b4b6e" />
            </marker>
            <marker
              id="arrowhead-highlight"
              markerWidth="8"
              markerHeight="6"
              refX="14"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 8 3, 0 6" fill="#6366f1" />
            </marker>
          </defs>

          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
            {/* 1. Render Directed Edges */}
            {subgraph?.edges.map((edge) => {
              const srcNode = layoutNodes.find((n) => n.id === edge.source);
              const dstNode = layoutNodes.find((n) => n.id === edge.target);
              if (!srcNode || !dstNode) return null;

              const isConnectedToSelected =
                selectedNode &&
                (selectedNode.id === srcNode.id || selectedNode.id === dstNode.id);

              return (
                <line
                  key={edge.id}
                  x1={srcNode.x}
                  y1={srcNode.y}
                  x2={dstNode.x}
                  y2={dstNode.y}
                  stroke={isConnectedToSelected ? '#818cf8' : '#232f48'}
                  strokeWidth={isConnectedToSelected ? 2 : 1.2}
                  strokeDasharray={isConnectedToSelected ? undefined : undefined}
                  markerEnd={isConnectedToSelected ? 'url(#arrowhead-highlight)' : 'url(#arrowhead)'}
                  className="transition-colors duration-150"
                />
              );
            })}

            {/* 2. Render Graph Nodes */}
            {layoutNodes.map((node) => {
              const isSelected = selectedNode?.id === node.id;
              const isHovered = hoveredNode?.id === node.id;
              const color = getNodeColor(node.label, node.is_center);
              const glow = getNodeGlow(node.label, node.is_center);
              const radius = node.is_center ? 16 : 10;

              return (
                <g
                  key={node.id}
                  transform={`translate(${node.x}, ${node.y})`}
                  className="cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectNode(node);
                  }}
                  onMouseEnter={() => setHoveredNode(node)}
                  onMouseLeave={() => setHoveredNode(null)}
                >
                  {/* Outer Pulsing Aura for Center or Selected */}
                  {(node.is_center || isSelected) && (
                    <circle
                      r={radius + 8}
                      fill="none"
                      stroke={color}
                      strokeWidth={1.5}
                      strokeOpacity={0.6}
                      className="animate-ping"
                      style={{ animationDuration: '3s' }}
                    />
                  )}

                  {/* Selection Ring */}
                  {isSelected && (
                    <circle
                      r={radius + 5}
                      fill="none"
                      stroke="#ffffff"
                      strokeWidth={2}
                      strokeDasharray="3 3"
                    />
                  )}

                  {/* Main Node Circle */}
                  <circle
                    r={radius}
                    fill={color}
                    stroke="#0b0e17"
                    strokeWidth={2}
                    style={{
                      filter: `drop-shadow(0 0 8px ${glow})`,
                    }}
                  />

                  {/* Inner center dot */}
                  <circle r={radius * 0.35} fill="#ffffff" fillOpacity={0.85} />

                  {/* Label tag */}
                  <text
                    y={radius + 14}
                    textAnchor="middle"
                    fill={node.is_center ? '#c7d2fe' : '#94a3b8'}
                    fontSize={node.is_center ? 11 : 9}
                    fontFamily="monospace"
                    className="font-medium pointer-events-none"
                  >
                    {node.tx_id.toString().slice(-6)}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* Floating Hover Tooltip */}
      {hoveredNode && (
        <div className="absolute top-4 left-4 z-30 pointer-events-none bg-lab-surface/95 border border-lab-borderLight p-3 rounded-lg shadow-2xl backdrop-blur-md text-xs font-mono space-y-1">
          <div className="flex items-center space-x-2">
            <span className="text-slate-400">Tx ID:</span>
            <span className="text-white font-bold">{hoveredNode.tx_id}</span>
            {hoveredNode.is_center && (
              <span className="px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300 text-[10px]">
                Target
              </span>
            )}
          </div>
          <div className="flex items-center justify-between space-x-4">
            <span className="text-slate-400">Class:</span>
            <span
              className={`font-semibold capitalize ${
                hoveredNode.label === 'illicit'
                  ? 'text-rose-400'
                  : hoveredNode.label === 'licit'
                  ? 'text-emerald-400'
                  : 'text-slate-400'
              }`}
            >
              {hoveredNode.label}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Timestep:</span>
            <span className="text-cyan-300">t = {hoveredNode.timestep} ({hoveredNode.split})</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Degree (in/out):</span>
            <span className="text-slate-200">
              {hoveredNode.in_degree} in / {hoveredNode.out_degree} out
            </span>
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {loading && (
        <div className="absolute inset-0 z-30 bg-black/60 backdrop-blur-sm flex items-center justify-center">
          <div className="flex flex-col items-center space-y-2 font-mono text-xs text-indigo-300">
            <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            <span>Extracting Local Neighborhood...</span>
          </div>
        </div>
      )}
    </div>
  );
};
