'use client';

type BadgeVariant = 'default' | 'success' | 'error' | 'warning' | 'info';

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-surface-active text-text-secondary',
  success: 'bg-green-500/15 text-green-400',
  error: 'bg-red-500/15 text-red-400',
  warning: 'bg-yellow-500/15 text-yellow-400',
  info: 'bg-blue-500/15 text-blue-400',
};

export function Badge({ children, variant = 'default', className = '' }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${variantStyles[variant]} ${className}`}
    >
      {children}
    </span>
  );
}
