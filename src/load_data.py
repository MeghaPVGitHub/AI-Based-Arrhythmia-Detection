import wfdb
import matplotlib.pyplot as plt

# Load the ECG signal (example: record 100)
p_signal, fields = wfdb.rdsamp('data/100')  # Unpack signal and metadata
annotation = wfdb.rdann('data/100', 'atr')  # Load annotations

# Display metadata and annotations
print("Record Metadata:", fields)
print("Annotations:", annotation.__dict__)

# Create a WFDB record object manually for plotting
record = wfdb.Record(
    p_signal=p_signal,
    fs=fields['fs'],
    sig_name=fields['sig_name'],
    units=fields['units'],  # Add the units from metadata
    n_sig=len(fields['sig_name'])
)

# Plot the ECG signal with annotations
wfdb.plot_wfdb(record=record, annotation=annotation, title="ECG Signal - Record 100")
