package policy

import (
    "context"
    "errors"
    "os"

    "github.com/open-policy-agent/opa/rego"
)

// OPAEngine wraps a prepared Rego query for evaluating decisions.
// Expected policy to set data.shieldx.authz.decision to one of: "allow","deny","divert","tarpit".
type OPAEngine struct {
    prepared rego.PreparedEvalQuery
}

// LoadOPA compiles the given Rego file and prepares a query for decision.
func LoadOPA(path string) (*OPAEngine, error) {
    if path == "" {
        return nil, nil
    }
    if _, err := os.Stat(path); err != nil {
        return nil, err
    }
    ctx := context.Background()
    r := rego.New(
        rego.Query("data.shieldx.authz.decision"),
        rego.Load([]string{path}, nil),
    )
    pq, err := r.PrepareForEval(ctx)
    if err != nil {
        return nil, err
    }
    return &OPAEngine{prepared: pq}, nil
}

// Evaluate returns a policy Action and true if OPA produced a decision. If false, caller should fall back.
func (e *OPAEngine) Evaluate(input map[string]any) (Action, bool, error) {
    if e == nil {
        return "", false, nil
    }
    ctx := context.Background()
    rs, err := e.prepared.Eval(ctx, rego.EvalInput(input))
    if err != nil {
        return "", false, err
    }
    if len(rs) == 0 || len(rs[0].Expressions) == 0 {
        return "", false, nil
    }
    v := rs[0].Expressions[0].Value
    s, ok := v.(string)
    if !ok || s == "" {
        return "", false, nil
    }
    switch Action(s) {
    case ActionAllow, ActionDeny, ActionDivert, ActionTarpit:
        return Action(s), true, nil
    default:
        return "", false, errors.New("unsupported decision from OPA")
    }
}



