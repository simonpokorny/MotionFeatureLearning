import os
import pickle
import glob


def merge_metadata(input_path):
    """
    Merge individual look-up table and flows mins and maxs and store it in the input_path with the name metadata
    :param input_path: Path with the local metadata in the form metadata_[tfRecordName]
    """
    look_up_table = []
    flows_info = None

    os.chdir(input_path)
    for file in glob.glob("metadata_*"):
        file_name = os.path.abspath(file)
        try:
            with open(file_name, 'rb') as metadata_file:
                metadata_local = pickle.load(metadata_file)
                look_up_table.extend(metadata_local['look_up_table'])
                flows_information = metadata_local['flows_information']
                if flows_info is None:
                    flows_info = flows_information
                else:
                    flows_info['min_vx'] = min(flows_info['min_vx'], flows_information['min_vx'])
                    flows_info['min_vx'] = min(flows_info['min_vx'], flows_information['min_vy'])
                    flows_info['min_vz'] = min(flows_info['min_vz'], flows_information['min_vz'])
                    flows_info['max_vx'] = max(flows_info['max_vx'], flows_information['max_vx'])
                    flows_info['max_vy'] = max(flows_info['max_vy'], flows_information['max_vy'])
                    flows_info['max_vz'] = max(flows_info['max_vz'], flows_information['max_vz'])

        except FileNotFoundError:
            raise FileNotFoundError(
                "Metadata not found when merging individual metadata")

    # Save metadata into disk
    metadata = {'look_up_table': look_up_table,
                'flows_information': flows_info}
    with open(os.path.join(input_path, "metadata"), 'wb') as metadata_file:
        pickle.dump(metadata, metadata_file)


if __name__ == "__main__":
    merge_metadata(input_path="/home/pokorsi1/data/waymo_flow/preprocess/valid")