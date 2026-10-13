"""Conservative, horizon-specific gates over recorded surrogate evidence."""


def validated_objective_horizons(report):
    """Return horizons with complete trajectory, kinematic and gradient evidence.

    A global failure remains a global failure. This function only identifies
    explicitly scoped objectives for independent physical verification; it
    never enables unattended optimization or certifies full dynamics fidelity.
    """
    try:
        results = report['results']
        if set(results)!={'validation','test'}:
            return []
        valid = None
        for result in results.values():
            ids = [c['id'] for c in result['cases']]
            derivative = result['derivatives']
            kin = derivative['kinematics']
            if (len(ids)<3 or len(set(ids))!=len(ids) or not result['trajectory_gate_passed']
                or not derivative['kinematic_gate_passed'] or sorted(k['id'] for k in kin)!=sorted(ids)
                or not all(k['passed'] for k in kin)):
                return []
            candidates = set(g['horizon_s'] for g in derivative['cases'])
            horizons = set()
            for h in candidates:
                rows = [g for g in derivative['cases'] if g['horizon_s']==h]
                if sorted(g['id'] for g in rows)==sorted(ids) and all(g['passed'] for g in rows):
                    horizons.add(h)
            valid = horizons if valid is None else valid&horizons
        return sorted(valid or [])
    except (KeyError,TypeError,ValueError):
        return []
