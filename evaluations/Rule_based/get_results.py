import os
from google.cloud import storage
from tqdm import tqdm

def download_folder_from_bucket(bucket_name, source_folder, destination_folder, credentials_file):

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    blobs = bucket.list_blobs(prefix=source_folder)

    for blob in tqdm(blobs):
        if blob.name.endswith("/"):
            continue

        # construct the local path
        relative_path = os.path.relpath(blob.name, source_folder)
        local_path = os.path.join(destination_folder, relative_path)

        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        print(f"Downloading {blob.name} to {local_path}...")
        blob.download_to_filename(local_path)

    print("Download completed.")

if __name__ == "__main__":
    BUCKET_NAME = "nlp_result"
    #Single
    #SOURCE_FOLDER = "Results_v3/Single_Agent/Output"
    #DESTINATION_FOLDER = "./Results/Single_Result"
    #Multi_discussion
    #SOURCE_FOLDER = "Results_v3/SciFi-Dis/Output/multi_agent/SciFi-Dis_multi_teacher_roleplay_3_5_meta-llama"
    #DESTINATION_FOLDER = "./Results/Multidis_Result"
    #Multi_teacher
    #SOURCE_FOLDER = "Results_v4/SciFi-Teacher/Output/multi_agent/SciFi-Teacher_multi_teacher_roleplay_4_5_meta-llama"
    #DESTINATION_FOLDER = "./Results/Multiteacher_Result"
    #Multi_review
    #SOURCE_FOLDER = "Results_v4/SciFi-Review/Output/multi_agent/SciFi-Review_multi_teacher_roleplay_3_5_meta-llama"
    #DESTINATION_FOLDER = "./Results/Multireview_Result"
    #Decode(Rep_Pen)
    #SOURCE_FOLDER = "Results_v3/Decoded_Exp/Rep_Pen/SciFi-Teacher/Output/multi_agent/SciFi-Teacher_multi_teacher_roleplay_4_5_meta-llama"
    #DESTINATION_FOLDER = "./Results/Decoding/Rep_pen"
    #Decode(Top_k)
    #SOURCE_FOLDER = "Results_v3/Decoded_Exp/Top_k/SciFi-Teacher/Output/multi_agent/SciFi-Teacher_multi_teacher_roleplay_4_5_meta-llama"
    #DESTINATION_FOLDER = "./Results/Decoding/Top_k"
    #Decode(Top_p)
    #SOURCE_FOLDER = "Results_v3/Decoded_Exp/Top_p/SciFi-Teacher/Output/multi_agent/SciFi-Teacher_multi_teacher_roleplay_4_5_meta-llama"
    #DESTINATION_FOLDER = "./Results/Decoding/Top_p"
    #Decode(Temp)
    SOURCE_FOLDER = "Results_v3/Decode_Exp/Temp/SciFi-Teacher/Output/multi_agent/SciFi-Teacher_multi_teacher_roleplay_4_5_meta-llama"
    DESTINATION_FOLDER = "./Results/Decoding/Temp"
    CREDENTIALS_FILE = "./Evaluation/Rule_based/secrets.json"

    download_folder_from_bucket(BUCKET_NAME, SOURCE_FOLDER, DESTINATION_FOLDER, CREDENTIALS_FILE)
