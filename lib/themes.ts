export type Theme = 'dark' | 'light' | 'high-contrast'

export interface ThemeConfig {
  name: string
  value: Theme
  colors: {
    background: string
    foreground: string
    editor: {
      background: string
      text: string
      lineNumber: string
      cursor: string
      selection: string
      token: {
        keyword: string
        string: string
        number: string
        comment: string
        function: string
      }
    }
  }
}

export const themes: Record<Theme, ThemeConfig> = {
  dark: {
    name: 'Dark',
    value: 'dark',
    colors: {
      background: '#1e1e1e',
      foreground: '#e0e0e0',
      editor: {
        background: '#1e1e1e',
        text: '#d4d4d4',
        lineNumber: '#858585',
        cursor: '#aeafad',
        selection: '#264f78',
        token: {
          keyword: '#569cd6',
          string: '#ce9178',
          number: '#b5cea8',
          comment: '#6a9955',
          function: '#dcdcaa',
        },
      },
    },
  },
  light: {
    name: 'Light',
    value: 'light',
    colors: {
      background: '#ffffff',
      foreground: '#333333',
      editor: {
        background: '#ffffff',
        text: '#000000',
        lineNumber: '#999999',
        cursor: '#000000',
        selection: '#add6ff',
        token: {
          keyword: '#0000ff',
          string: '#a31515',
          number: '#098658',
          comment: '#008000',
          function: '#795e26',
        },
      },
    },
  },
  'high-contrast': {
    name: 'High Contrast',
    value: 'high-contrast',
    colors: {
      background: '#000000',
      foreground: '#ffffff',
      editor: {
        background: '#000000',
        text: '#ffffff',
        lineNumber: '#cccccc',
        cursor: '#ffffff',
        selection: '#ffffff33',
        token: {
          keyword: '#ffff00',
          string: '#ff0000',
          number: '#00ff00',
          comment: '#00ffff',
          function: '#ff00ff',
        },
      },
    },
  },
}

export function getThemeConfig(theme: Theme): ThemeConfig {
  return themes[theme]
}

export function getMonacoTheme(theme: Theme) {
  const config = themes[theme]
  
  if (theme === 'dark') {
    return 'vs-dark'
  } else if (theme === 'light') {
    return 'vs'
  } else {
    return 'hc-black'
  }
}
