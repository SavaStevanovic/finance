import pyreadstat
import json
from datetime import datetime
file_path = 'output_file.json'

df, meta = pyreadstat.read_sas7bdat('kidpan_data.sas7bdat', catalog_file='formats.sas7bcat', formats_as_category=True, formats_as_ordered_category=False)


def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")

# Dump the dictionary into the file using the custom serialization function
with open(file_path, 'w') as json_file:
    json.dump(meta.__dict__, json_file, default=serialize_datetime, indent=4)
df
