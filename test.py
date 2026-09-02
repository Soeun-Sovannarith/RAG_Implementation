import os
from vector import parse_file_to_documents

def test_load_all_files_in_data():
  data_dir = "data"
  if not os.path.exists(data_dir):
    print(f"Directory '{data_dir}' does not exist.")
    return

  all_files = [f for f in os.listdir(data_dir) if not f.startswith(".")]
  print(f"Found {len(all_files)} file(s) in '{data_dir}/': {all_files}\n")

  for file_name in sorted(all_files):
    file_path = os.path.join(data_dir, file_name)
    docs = parse_file_to_documents(file_path)

    if docs:
      print(f"File: '{file_path}' ({len(docs)} document record(s)):")
      print(f"   Standardized Output Sample:\n   {docs[0]}\n")
      print("-" * 65)

if __name__ == "__main__":
  test_load_all_files_in_data()
