// Controlled demo inputs. The architecture is real; these stand in for live carrier data.
// All numbers are fictional.

export const SCENARIOS = [
  {
    id: 'legit-bank',
    label: 'Legitimate call from your bank',
    call: {
      number: '+12125550142',
      claimedOrg: 'First National Bank',
      claimedLocal: true,
      verstat: 'TN-Validation-Passed-A',
      originCountry: 'US',
      user: { country: 'US', region: 'New York' },
    },
    ipqs: {
      success: true, valid: true, active: true, fraud_score: 4, VOIP: false, prepaid: false,
      risky: false, recent_abuse: false, spammer: false, line_type: 'Landline',
      carrier: 'Verizon', country: 'US', region: 'New York', city: 'New York',
    },
    transcript:
      "Hello, this is Maria from First National Bank's card services. We're calling to confirm a purchase at a hardware store for 84 dollars. " +
      "If it wasn't you, please hang up and call the number on the back of your card. We will never ask for your PIN or full card number.",
  },
  {
    id: 'grandparent',
    label: 'Grandparent scam (spoofed, overseas)',
    call: {
      // Shows the victim's own area code (212) but the carrier registered it in Florida: neighbor spoofing.
      number: '+12125550187',
      claimedOrg: 'Grandchild',
      claimedLocal: true,
      verstat: 'TN-Validation-Passed-C',
      originCountry: 'NG',
      user: { country: 'US', region: 'New York' },
    },
    ipqs: {
      success: true, valid: true, active: true, fraud_score: 91, VOIP: true, prepaid: true,
      risky: true, recent_abuse: true, spammer: true, line_type: 'VOIP',
      carrier: 'Bandwidth VoIP Reseller', country: 'US', region: 'Florida', city: 'Miami',
    },
    transcript:
      "Grandma? It's me. Please don't tell Mom and Dad, I'm in big trouble. I was in a car accident and I got arrested. " +
      "I need bail money right now. A lawyer is going to call you. You have to buy gift cards and read me the numbers, " +
      "or I could go to jail tonight. Please hurry, and don't tell anyone, I'm so embarrassed.",
  },
  {
    id: 'tech-support',
    label: 'Tech support scam (VoIP, toll-free)',
    call: {
      number: '+18885550133',
      claimedOrg: 'Microsoft Support',
      claimedLocal: false,
      verstat: 'TN-Validation-Passed-B',
      originCountry: 'IN',
      user: { country: 'US', region: 'California' },
    },
    ipqs: {
      success: true, valid: true, active: true, fraud_score: 82, VOIP: true, prepaid: false,
      risky: true, recent_abuse: true, spammer: false, line_type: 'Toll Free',
      carrier: 'Twilio', country: 'US', region: 'Texas', city: 'Dallas',
    },
    transcript:
      "This is Microsoft Windows technical support. Your computer has been sending us error messages and we detected a virus and hackers on your network. " +
      "I need you to install remote access software so I can fix it. Go to the website I will give you and enter the code. " +
      "The fee is 299 dollars, you can pay with gift cards or bitcoin. Do not turn off your computer or it will crash.",
  },
];

export function getScenario(id) {
  return SCENARIOS.find((s) => s.id === id) || null;
}
