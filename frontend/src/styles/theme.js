const colors = {
  primary: '#ec4899',
  primaryDark: '#db2777',
  primaryLight: 'rgba(236, 72, 153, 0.1)',
  background: '#000000',
  text: '#ffffff',
  textSecondary: 'rgba(255, 255, 255, 0.7)'
};

const spacing = {
  xs: '8px',
  sm: '12px',
  md: '16px',
  lg: '24px',
  xl: '32px'
};

const borderRadius = {
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '32px'
};

const shadows = {
  card: '0 20px 40px rgba(0, 0, 0, 0.2)',
  hover: '0 10px 20px rgba(236, 72, 153, 0.2)'
};

const typography = {
  h1: {
    fontSize: '36px',
    fontWeight: '600',
    lineHeight: '1.2'
  },
  h2: {
    fontSize: '18px',
    fontWeight: '500',
    lineHeight: '1.3'
  },
  body: {
    fontSize: '14px',
    lineHeight: '1.5'
  }
};

export const theme = {
  colors,
  spacing,
  borderRadius,
  shadows,
  typography
};

export const styles = {
  container: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    backgroundColor: colors.background,
    padding: spacing.xl
  },
  card: {
    width: '100%',
    maxWidth: '600px',
    textAlign: 'center'
  },
  heading: {
    ...typography.h1,
    color: colors.text,
    marginBottom: spacing.xl
  },
  searchInput: {
    width: '100%',
    padding: `${spacing.md} ${spacing.lg}`,
    borderRadius: '8px',
    border: 'none',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    color: colors.text,
    fontSize: typography.body.fontSize,
    transition: 'all 0.3s ease',
    '&:focus': {
      outline: 'none',
      backgroundColor: 'rgba(255, 255, 255, 0.15)'
    }
  },
  button: {
    backgroundColor: colors.primary,
    color: colors.text,
    padding: `${spacing.md} ${spacing.lg}`,
    width: '100%',
    marginTop: spacing.md,
    borderRadius: '8px',
    border: 'none',
    cursor: 'pointer',
    fontSize: typography.body.fontSize,
    fontWeight: '500',
    transition: 'all 0.3s ease',
    '&:hover': {
      backgroundColor: colors.primaryDark
    }
  },
  uploadSection: {
    marginTop: spacing.xl,
    padding: spacing.lg,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '8px'
  },
  uploadTitle: {
    ...typography.h2,
    color: colors.textSecondary,
    marginBottom: spacing.md
  },
  fileInput: {
    width: '100%',
    padding: spacing.md,
    border: 'none',
    borderRadius: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    color: colors.text,
    cursor: 'pointer',
    fontSize: typography.body.fontSize
  },
  resultsSection: {
    marginTop: spacing.xl,
    padding: spacing.lg,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '8px'
  },
  resultsTitle: {
    ...typography.h2,
    color: colors.textSecondary,
    marginBottom: spacing.lg
  },
  bestMatch: {
    marginBottom: spacing.lg,
    padding: spacing.lg,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '8px'
  },
  bestMatchTitle: {
    ...typography.h2,
    color: colors.textSecondary,
    marginBottom: spacing.md
  },
  imageContainer: {
    width: '100%',
    marginTop: spacing.md,
    borderRadius: '8px',
    overflow: 'hidden'
  },
  memeImage: {
    width: '100%',
    maxHeight: '300px',
    objectFit: 'cover',
    borderRadius: '8px'
  },
  similarity: {
    marginTop: spacing.sm,
    color: colors.textSecondary,
    fontSize: typography.body.fontSize
  },
  similarTitle: {
    ...typography.h2,
    color: colors.textSecondary,
    marginBottom: spacing.md
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: spacing.md
  },
  gridItem: {
    padding: spacing.md,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '8px'
  }
}; 