"""Prospective identifiers for the distinct learning-effect confirmation."""
def design():
    return dict(schema='effect-confirmation-1', reset_seed=82800, contexts=[
        dict(id=f'confirm_{level}_{wait}_{start}', split='test', level=level, wait=wait, start=start)
        for level in ['1-1','2-1','3-1','4-1'] for wait in [9,25,41,57]
        for start in [77,85,93,101,109,117,125,133]])
