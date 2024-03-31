## Downstream preprocessing 
import os
import pandas as pd  
import mne  # For EEG data processing
import numpy as np

categories = {"ELECTRONICS": ["MICROWAVE", "TELEVISION", "PRINTER", "SCANNER", "TABLET", "SPEAKER", "RADIO", "STEREO", "COMPUTER", "HEADPHONES", "PHONE", "CALCULATOR"], "FABRIC": ["NYLON", "COTTON", "LEATHER", "DENIM", "WOOL", "SILK", "FLEECE", "SUEDE", "VELVET", "SPANDEX", "LINEN", "SATIN"], "VEGETABLES": ["CARROT", "CUCUMBER", "CELERY", "POTATO", "ONION", "CAULIFLOWER", "SPINACH", "BROCCOLI", "CORN", "CABBAGE", "PEAS", "LETTUCE"], "APPLIANCES": ["FREEZER", "VACUUM", "TOASTER", "REFRIGERATOR", "DRYER", "BLENDER", "IRON", "WASHER", "MIXER", "DISHWASHER", "OVEN", "STOVE"], "KITCHENTOOLS": ["GRATER", "STRAINER", "COLANDER", "CORKSCREW", "FORK", "KNIFE", "SPOON", "WHISK", "LADLE", "PEELER", "TONGS", "SPATULA"], "DESSERTS": ["CHEESECAKE", "CAKE", "PIE", "COOKIE", "COBBLER", "MOUSSE", "PUDDING", "BROWNIES", "PASTRY", "JELLO", "CUPCAKE", "FUDGE"], "FURNITURE": ["DESK", "TABLE", "RECLINER", "CHAIR", "BED", "BOOKCASE", "DRESSER", "CABINET", "NIGHTSTAND", "SOFA", "SHELF", "COUCH"], "WEATHER": ["FOG", "TORNADO", "RAIN", "CLOUD", "LIGHTNING", "HURRICANE", "THUNDER", "SNOW", "THUNDERSTORM", "HUMIDITY", "WIND", "DRIZZLE"], "INSTRUMENTS": ["ORGAN", "TROMBONE", "GUITAR", "VIOLIN", "DRUM", "BASS", "CELLO", "FLUTE", "PIANO", "BANJO", "TUBA", "TRUMPET"], "CLOTHING": ["BLOUSE", "DRESS", "PANTS", "UNDERWEAR", "SHORTS", "SHIRT", "SOCKS", "COAT", "JEANS", "SHOES", "JACKET", "SKIRT"], "BUILDING": ["DOOR", "ATTIC", "WALL", "FLOOR", "CEILING", "ROOF", "WINDOW", "BATHROOM", "HALLWAY", "ROOM", "STAIRS", "FOUNDATION"], "LANDSCAPES": ["LAKE", "STREAM", "FOREST", "VALLEY", "OCEAN", "CREEK", "BEACH", "WOODS", "MOUNTAIN", "RIVER", "ISLAND", "HILLS"], "TOOLS": ["DRILL", "SCREWDRIVER", "WRENCH", "CROWBAR", "PLIERS", "AXE", "LEVEL", "SAW", "HAMMER", "SHOVEL", "SANDER", "RATCHET"], "OCEANANIMALS": ["CRAB", "OCTOPUS", "TURTLE", "OYSTER", "DOLPHIN", "WHALE", "LOBSTER", "JELLYFISH", "SHRIMP", "PLANKTON", "SQUID", "FISH"], "ZOO": ["GIRAFFE", "CHIMPANZEE", "BEAR", "ORANGUTAN", "LION", "TIGER", "GORILLA", "MONKEY", "ELEPHANT", "CHEETAH", "ZEBRA", "PANDA"], "VEHICLES": ["BUS", "CAR", "TRUCK", "JEEP", "BIKE", "VAN", "BOAT", "TRAIN", "MOTORCYCLE", "SCOOTER", "PLANE", "MOPED"], "BEVERAGES": ["TEA", "COLA", "RUM", "WATER", "JUICE", "COFFEE", "CIDER", "WINE", "MILK", "BEER", "SODA", "LEMONADE"], "FLOWERS": ["VIOLET", "ROSE", "ORCHID", "LILY", "SUNFLOWER", "CARNATION", "DAISY", "PETUNIA", "LILAC", "MARIGOLD", "DAFFODIL", "TULIP"], "OFFICESUPPLIES": ["ERASER", "TAPE", "MARKER", "BINDER", "PAPER", "PEN", "RULER", "PENCIL", "STAPLER", "FOLDER", "GLUE", "NOTEBOOK"], "BIRDS": ["HAWK", "SPARROW", "CROW", "SEAGULL", "CARDINAL", "FINCH", "BLUEBIRD", "DOVE", "GOOSE", "PIGEON", "ROBIN", "DUCK"], "FRUIT": ["PLUM", "GRAPE", "LIME", "PEAR", "BLUEBERRY", "APPLE", "STRAWBERRY", "LEMON", "BANANA", "CHERRY", "ORANGE", "PEACH"], "INSECTS": ["BEETLE", "FLY", "BEE", "MOTH", "SPIDER", "MOSQUITO", "BUTTERFLY", "GNAT", "ANT", "ROACH", "LADYBUG", "CRICKET"], "TREES": ["MAPLE", "CEDAR", "SPRUCE", "EVERGREEN", "OAK", "HICKORY", "ELM", "PINE", "CYPRESS", "BIRCH", "WALNUT", "WILLOW"], "FARMANIMALS": ["PIG", "CALF", "COW", "TURKEY", "HEN", "ROOSTER", "LAMB", "BULL", "CHICKEN", "HORSE", "GOAT", "SHEEP"], "BODYPARTS": ["NOSE", "HEAD", "HAND", "LEG", "FOOT", "TOE", "FINGERS", "EYE", "EAR", "ELBOW", "KNEE", "ARM"], "TOYS": ["PUZZLE", "FOOTBALL", "DOLLHOUSE", "BARBIE", "FRISBEE", "BLOCKS", "BASEBALL", "BASKETBALL", "BALL", "SLINKY", "DOLL", "MARBLES"]}
valid_labels = {label for category_labels in categories.values() for label in category_labels}
print(valid_labels)
chunk_number = 0

def preprocess_eeg(eeg_data, sfreq, patch_length, stride):
    """
    Preprocess EEG data by applying FFT, normalizing, and extracting sliding window patches.
    
    Args:
        eeg_data: Numpy array of EEG data.
        sfreq: Sampling frequency of the EEG data.
        patch_length: The length of each patch after splitting.
        stride: The stride with which to apply the sliding window for patch extraction.
        
    Returns:
        A numpy array of preprocessed data.
    """
    # Apply FFT
    fft_data = np.abs(np.fft.fft(eeg_data, axis=1))
    # Normalize the FFT data by channel
    normalized_data = (fft_data - np.mean(fft_data, axis=1, keepdims=True)) / np.std(fft_data, axis=1, keepdims=True)
    
    patches = []
    # Extract sliding window patches for each channel
    for channel in normalized_data:
        all_patches = np.lib.stride_tricks.sliding_window_view(channel, window_shape=(patch_length,))
        stride_indices = np.arange(0, all_patches.shape[0], stride)
        patches.append(all_patches[stride_indices])
    preprocessed_data = np.concatenate(patches, axis=0)
    return preprocessed_data

def read_labels(labels_file):
    """
    Reads a TSV file containing labels and returns a DataFrame.
    
    Args:
        labels_file: Path to the .tsv file containing labels.
        
    Returns:
        DataFrame with columns ['start_time', 'duration', 'label']
    """
    return pd.read_csv(labels_file, sep='\t')

def match_labels_to_data(eeg_data, raw, labels_df):
    """
    Matches labels to EEG data segments based on timing information and filters out segments without a valid label.
    """
    valid_segments = []
    sfreq = raw.info['sfreq']
    for _, row in labels_df.iterrows():
        # Check if the label is a string and not NaN or any float value
        if isinstance(row['label'], str):
            label = row['label'].upper()  # Now it's safe to call .upper()
            # Proceed if the label is in our list of valid labels
            if label in valid_labels:
                start_sample = int(row['start_time'] * sfreq)
                end_sample = start_sample + int(row['duration'] * sfreq)
                valid_segments.append((start_sample, end_sample, label))
                print(f"Found valid segment with label: {label}")
                print(valid_segments[-1])
        else:
            continue  # Skip this row if the label is not a string

    return valid_segments


def save_chunks_with_labels(preprocessed_data, label, chunk_size, save_dir, file_prefix):
    global chunk_number
    """
    Saves the preprocessed data and labels in chunks to the specified directory, correctly handling string labels.

    Args:
        preprocessed_data: The data to be saved.
        labels: The labels associated with each data patch.
        chunk_size: The size of each chunk.
        save_dir: Directory where chunks will be saved.
        file_prefix: Prefix for the chunk filenames.
    """
    num_chunks = int(np.ceil(preprocessed_data.shape[0] / chunk_size))
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        data_chunk = preprocessed_data[start_idx:end_idx]
        labels_chunk = label

        # Pad the last chunk if it's smaller than the chunk size for data
        if data_chunk.shape[0] < chunk_size:
            data_padding = ((0, chunk_size - data_chunk.shape[0]), (0, 0))
            data_chunk = np.pad(data_chunk, data_padding, 'constant')

        # Save each chunk to a file
        chunk_filepath = os.path.join(save_dir, f"{file_prefix}_chunk_{chunk_number}.npz")
        print(data_chunk.shape, np.array(labels_chunk))
        print(f"Saving chunk to: {chunk_filepath}")
        np.savez_compressed(chunk_filepath, data=data_chunk, labels=np.array(labels_chunk))
        chunk_number += 1



def preprocess_eeg_and_labels(folder_path, patch_length=128, stride=64, chunk_size=128, save_dir='./preprocessed_chunks_labeled'):
    """
    Preprocess EEG data and save along with matched labels, ensuring only data with valid labels are saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for foldername in os.listdir(folder_path):
        if foldername.startswith("sub"):
            for session in ["ses-0", "ses-1"]:
                subject_dir = os.path.join(folder_path, foldername, session, "ieeg")
                if os.path.isdir(subject_dir):
                    for filename in os.listdir(subject_dir):
                        if filename.endswith(".edf"):
                            filepath = os.path.join(subject_dir, filename)
                            constantname = filename.split('catFR1')[0] + 'catFR1_beh.edf'
                            labels_df = read_labels(os.path.join(folder_path, foldername, session, "beh", f"{constantname.split('.')[0]}_filtered.tsv"))
                            print(labels_df)
                            raw = mne.io.read_raw_edf(filepath, preload=True)
                            eeg_data = raw.get_data()
                            valid_segments = match_labels_to_data(eeg_data, raw, labels_df)
                            
                            for start_sample, end_sample, label in valid_segments:
                                print(f"Processing segment with label: {label}")
                                segment_data = eeg_data[:, start_sample:end_sample]
                                preprocessed = preprocess_eeg(segment_data, raw.info['sfreq'], patch_length, stride)

                                save_chunks_with_labels(preprocessed, [label], chunk_size, save_dir, filename.split('.')[0])


folder_path = ".\data"
preprocess_eeg_and_labels(folder_path)

