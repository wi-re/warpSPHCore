/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  wiki: [
    'index',
    'quickstart',
    {
      type: 'category',
      label: 'Operators',
      items: [
        'operations/density',
        'operations/gradient',
        'operations/divergence',
        'operations/curl',
        'operations/laplacian',
        'operations/interpolate',
        'operations/covariance',
      ],
    },
    {
      type: 'category',
      label: 'CRK corrections',
      items: [
        'crk/crk-overview',
        'crk/crk-moments',
        'crk/crk-volume',
        'crk/crk-density',
      ],
    },
    'kernels',
    'neighbor-search',
    'renorm',
    'autograd',
    'data-types',
    'api',
  ],
};

export default sidebars;
