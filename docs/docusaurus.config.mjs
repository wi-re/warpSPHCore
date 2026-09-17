import { fileURLToPath } from 'node:url';
import { themes as prismThemes } from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'warpSPHCore',
  tagline:
    'Warp + PyTorch SPH core operators — differentiable density, gradients, Laplacians, CRK corrections',
  favicon: 'img/favicon.svg',
  organizationName: 'wi-re',
  projectName: 'warpSPHCore',

  // The account's custom domain serves the repo under this base path.
  url: 'https://fluids.dev',
  baseUrl: '/warpSPHCore/',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',
  onBrokenAnchors: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  themes: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        indexDocs: true,
        indexPages: true,
        indexBlog: false,
        docsRouteBasePath: '/docs',
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: fileURLToPath(new URL('./sidebars.mjs', import.meta.url)),
          sidebarCollapsed: false,
          // Math rendering (KaTeX). remark-math registers a micromark
          // extension, so $...$ / $$...$$ spans are tokenized before MDX's
          // expression parser sees the LaTeX braces inside them.
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          editUrl: ({ docPath }) =>
            `https://github.com/wi-re/warpSPHCore/edit/main/docs/docs/${docPath}`,
        },
        blog: false,
        theme: {
          customCss: fileURLToPath(new URL('./src/css/custom.css', import.meta.url)),
        },
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'light',
      },

      navbar: {
        title: 'warpSPHCore',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'wiki',
            position: 'left',
            label: 'Wiki',
          },
          { to: 'docs/kernels', position: 'left', label: 'Kernels' },
          { to: 'docs/autograd', position: 'left', label: 'Autodiff' },
          {
            to: 'https://github.com/wi-re/warpSPHCore',
            label: 'GitHub',
            position: 'right',
            className: 'header-github-link',
          },
        ],
      },

      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              { label: 'Wiki', to: 'docs' },
              { label: 'Density', to: 'docs/operations/density' },
              { label: 'Gradient', to: 'docs/operations/gradient' },
              { label: 'Kernel library', to: 'docs/kernels' },
              { label: 'Autodiff machinery', to: 'docs/autograd' },
            ],
          },
          {
            title: 'Repository',
            items: [
              {
                label: 'GitHub',
                to: 'https://github.com/wi-re/warpSPHCore',
              },
              {
                label: 'README',
                to: 'https://github.com/wi-re/warpSPHCore/blob/main/README.md',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Rene Winchenbach. Built with Docusaurus.`,
      },

      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.githubDark,
      },

      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 3,
      },
    }),
};

export default config;
