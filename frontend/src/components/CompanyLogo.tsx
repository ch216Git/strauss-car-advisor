import logo from "../assets/strauss-logo.png";

// The Strauss group's logo, at the start of both screens' headers.
export function CompanyLogo() {
  return <img className="header__company" src={logo} alt="שטראוס" width={40} height={40} />;
}
