import { themes as prismThemes } from "prism-react-renderer"
import type { Config } from "@docusaurus/types"
import type * as Preset from "@docusaurus/preset-classic"

const config: Config = {
  title: "Codeflash Docs",
  tagline: "Code optimization is cool",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://docs.codeflash.ai",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  // organizationName: 'facebook', // Usually your GitHub org/user name.
  // projectName: 'docusaurus', // Usually your repo name.

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  scripts: [
    {
      src: "https://widget.intercom.io/widget/ljxo1nzr",
      async: true,
      onLoad: `window.Intercom('boot', {
        app_id: "ljxo1nzr"
      });`,
    },
    {
      src: "https://app.posthog.com/static/array.js",
      strategy: "afterInteractive",
      onLoad: `window.posthog = window.posthog || [];
               window.posthog.init("phc_aUO790jHd7z1SXwsYCz8dRApxueplZlZWeDSpKc5hol", {
                 api_host: "https://us.i.posthog.com",
               });`,
    },
  ],
  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          routeBasePath: "/",
          // Please change this to your repo.
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    colorMode: {
      defaultMode: "dark",
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    image: "img/codeflash_social_card.jpg",
    navbar: {
      // title: 'My Site',
      logo: {
        href: "https://codeflash.ai/",
        alt: "Codeflash Logo",
        src: "img/codeflash_light.svg",
        srcDark: "img/codeflash_darkmode.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Docs",
        },
        // {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: "https://app.codeflash.ai/",
          label: "Get Started",
          position: "right",
        },
      ],
    },
    docs : {
      sidebar: {
        autoCollapseCategories: false,
        hideable: false,
      }
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Navigation",
          items: [
            {
              label: "Home Page",
              to: "https://codeflash.ai/",
            },
            {
              label: "PyPI",
              to: "https://pypi.org/project/codeflash/",
            },
            {
              label: "Get Started",
              to: "https://app.codeflash.ai/",
            },
          ],
        },
        {
          title: "Get in Touch",
          items: [
            {
              label: "Careers",
              to: "mailto:careers@codeflash.ai",
            },
            {
              label: "contact@codeflash.ai",
              href: "mailto:contact@codeflash.ai",
            },
          ],
        },
      ],
      copyright: `©2024 CodeFlash Inc.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["bash", "toml"],
    },
    algolia: {
      // The application ID provided by Algolia
      appId: 'Y1C10T0Z7E',

      // Public API key: it is safe to commit it
      apiKey: '4d1d294b58eb97edec121c9c1c079c23',

      indexName: 'codeflash',

      // Optional: see doc section below
      contextualSearch: true,

      // Optional: Algolia search parameters
      searchParameters: {},

      // Optional: path for search page that enabled by default (`false` to disable it)
      searchPagePath: 'search',

      // Optional: whether the insights feature is enabled or not on Docsearch (`false` by default)
      insights: true,

      //... other Algolia params
    },

  } satisfies Preset.ThemeConfig,
}

export default config
