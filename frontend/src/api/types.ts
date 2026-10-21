// The public HTTP contract, mirroring backend/app/api/schemas.py. Change both together.

export const UI_STATES = [
  "idle",
  "eligibility_check",
  "not_eligible",
  "collecting",
  "estimate",
  "confirmed",
  "representative",
  "car_choice",
  "answered",
  "incomplete",
  "temporary_error",
] as const;

export type UiState = (typeof UI_STATES)[number];

export interface Car {
  car_id: string;
  manufacturer: string;
  commercial_name: string;
  registration_year: number;
  group: string;
  electric: boolean;
  family_track: boolean;
}

export interface Alternative {
  kind: "company_car" | "waiver";
  car?: Car;
}

export interface EligibilityFacts {
  occupation?: string;
  employee_group?: string;
  car_group?: string;
  choose_from?: string;
  min_children?: number;
  child_max_age?: number;
  eligible_children?: number;
}

export interface Eligibility {
  missing: string[];
  reason?: string;
  facts?: EligibilityFacts;
}

export interface Details {
  missing: string[];
  provided: string[];
}

export interface Side {
  kind: "company_car" | "waiver";
  car?: Car;
  net: number;
  use_value: number;
  participation: number;
  participation_source?: "stated" | "catalogue_default";
  maintenance: number;
  family_share?: number;
}

export interface Change {
  item: "use_value" | "maintenance" | "participation" | "family_share";
  current: number;
  alternative: number;
}

export interface ElectricBenefit {
  net: number;
  grossed_up: boolean;
  applies_to: "alternative" | "current";
}

export interface Comparison {
  status: "estimate" | "confirmed";
  estimated_fields: string[];
  current: Side;
  alternative: Side;
  net_difference: number;
  changes: Change[];
  electric_benefit?: ElectricBenefit;
}

export type RepTopic = "alternative_participation" | "current_car_not_in_catalogue" | "other";

export interface RepresentativeQuestion {
  question_id: string;
  topic: RepTopic;
  status: "open" | "answered";
  answer?: string; // the representative's own words, once answered
}

export interface Representative {
  questions: RepresentativeQuestion[];
  latest?: { question_id: string; topic: RepTopic; already_open: boolean };
}

export interface CarChoice {
  target: "current" | "alternative";
  options: Car[];
}

export interface ChatResponse {
  session_id: string;
  turn: number;
  reply: string;
  ui_state: UiState;
  retryable: boolean;
  alternative?: Alternative;
  current?: Car;
  eligibility?: Eligibility;
  details?: Details;
  comparison?: Comparison;
  representative?: Representative;
  car_choice?: CarChoice;
  resume_token?: string; // present on a committed turn; the browser keeps it for local history
}

export interface SessionCreated {
  session_id: string;
  ui_state: "idle";
}

export interface SessionRestored {
  session_id: string; // a new server id; the restored conversation continues under it
  turn: number;
  ui_state: "idle";
}

export interface ApiErrorBody {
  error: { code: string; message: string; fields?: string[] };
}

// --- The transport representative's screen ------------------------------------------------------

export interface QueueQuestion {
  session_id: string;
  question_id: string;
  topic: RepTopic;
  status: "open" | "answered";
  question: string;
  car?: Car;
  answer?: string;
  participation?: number;
}

export interface Queue {
  questions: QueueQuestion[];
}

export interface Catalogue {
  cars: Car[];
}

export interface AnswerRequest {
  session_id: string;
  question_id: string;
  answer: string;
  participation?: number;
  car_id?: string;
}
