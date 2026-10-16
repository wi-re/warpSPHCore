/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  wiki: [
    'index',
    'quickstart',
    'examples',
    'development-artifacts',
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
    'field-reconstruction-guide',
    'reconstruction',
    'mfm',
    'linalg',
    'lattice-calibration',
    'autograd',
    'data-types',
    'api',
  ],
};

export default sidebars;
