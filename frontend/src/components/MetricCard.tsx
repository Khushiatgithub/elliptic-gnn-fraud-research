import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  badge?: string;
  badgeColor?: 'emerald' | 'indigo' | 'cyan' | 'rose' | 'slate' | 'amber';
  icon?: React.ComponentType<{ className?: string }>;
  description?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subValue,
  badge,
  badgeColor = 'indigo',
  icon: Icon,
  description,
}) => {
  const badgeClasses = {
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    indigo: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
    cyan: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    rose: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
    slate: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  }[badgeColor];

  return (
    <div className="lab-card p-4 flex flex-col justify-between lab-card-hover group">
      <div className="flex items-start justify-between">
        <span className="text-xs font-mono uppercase tracking-wider text-lab-muted">
          {label}
        </span>
        {Icon && (
          <div className="p-1.5 rounded-md bg-lab-card border border-lab-border text-slate-400 group-hover:text-slate-200 transition-colors">
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="my-2">
        <div className="text-2xl font-bold text-white tracking-tight font-mono">
          {value}
        </div>
        {subValue && (
          <div className="text-xs font-mono text-slate-400 mt-0.5">
            {subValue}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between mt-1 pt-2 border-t border-lab-border/60">
        {description ? (
          <span className="text-[11px] text-lab-muted truncate">
            {description}
          </span>
        ) : (
          <span />
        )}

        {badge && (
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${badgeClasses}`}>
            {badge}
          </span>
        )}
      </div>
    </div>
  );
};
