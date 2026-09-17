const FeatureCard = ({ title, children }) => (
  <div className="col col--4">
    <div className="featureCard">
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  </div>
);

export default FeatureCard;
