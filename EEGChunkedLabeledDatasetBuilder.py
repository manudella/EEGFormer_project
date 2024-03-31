import torch
from torch.utils.data import IterableDataset
import os
import numpy as np
import math
from torch.utils.data import DataLoader
import os

semantic_schema = {"ELECTRONICS": ["MICROWAVE", "TELEVISION", "PRINTER", "SCANNER", "TABLET", "SPEAKER", "RADIO", "STEREO", "COMPUTER", "HEADPHONES", "PHONE", "CALCULATOR"], "FABRIC": ["NYLON", "COTTON", "LEATHER", "DENIM", "WOOL", "SILK", "FLEECE", "SUEDE", "VELVET", "SPANDEX", "LINEN", "SATIN"], "VEGETABLES": ["CARROT", "CUCUMBER", "CELERY", "POTATO", "ONION", "CAULIFLOWER", "SPINACH", "BROCCOLI", "CORN", "CABBAGE", "PEAS", "LETTUCE"], "APPLIANCES": ["FREEZER", "VACUUM", "TOASTER", "REFRIGERATOR", "DRYER", "BLENDER", "IRON", "WASHER", "MIXER", "DISHWASHER", "OVEN", "STOVE"], "KITCHENTOOLS": ["GRATER", "STRAINER", "COLANDER", "CORKSCREW", "FORK", "KNIFE", "SPOON", "WHISK", "LADLE", "PEELER", "TONGS", "SPATULA"], "DESSERTS": ["CHEESECAKE", "CAKE", "PIE", "COOKIE", "COBBLER", "MOUSSE", "PUDDING", "BROWNIES", "PASTRY", "JELLO", "CUPCAKE", "FUDGE"], "FURNITURE": ["DESK", "TABLE", "RECLINER", "CHAIR", "BED", "BOOKCASE", "DRESSER", "CABINET", "NIGHTSTAND", "SOFA", "SHELF", "COUCH"], "WEATHER": ["FOG", "TORNADO", "RAIN", "CLOUD", "LIGHTNING", "HURRICANE", "THUNDER", "SNOW", "THUNDERSTORM", "HUMIDITY", "WIND", "DRIZZLE"], "INSTRUMENTS": ["ORGAN", "TROMBONE", "GUITAR", "VIOLIN", "DRUM", "BASS", "CELLO", "FLUTE", "PIANO", "BANJO", "TUBA", "TRUMPET"], "CLOTHING": ["BLOUSE", "DRESS", "PANTS", "UNDERWEAR", "SHORTS", "SHIRT", "SOCKS", "COAT", "JEANS", "SHOES", "JACKET", "SKIRT"], "BUILDING": ["DOOR", "ATTIC", "WALL", "FLOOR", "CEILING", "ROOF", "WINDOW", "BATHROOM", "HALLWAY", "ROOM", "STAIRS", "FOUNDATION"], "LANDSCAPES": ["LAKE", "STREAM", "FOREST", "VALLEY", "OCEAN", "CREEK", "BEACH", "WOODS", "MOUNTAIN", "RIVER", "ISLAND", "HILLS"], "TOOLS": ["DRILL", "SCREWDRIVER", "WRENCH", "CROWBAR", "PLIERS", "AXE", "LEVEL", "SAW", "HAMMER", "SHOVEL", "SANDER", "RATCHET"], "OCEANANIMALS": ["CRAB", "OCTOPUS", "TURTLE", "OYSTER", "DOLPHIN", "WHALE", "LOBSTER", "JELLYFISH", "SHRIMP", "PLANKTON", "SQUID", "FISH"], "ZOO": ["GIRAFFE", "CHIMPANZEE", "BEAR", "ORANGUTAN", "LION", "TIGER", "GORILLA", "MONKEY", "ELEPHANT", "CHEETAH", "ZEBRA", "PANDA"], "VEHICLES": ["BUS", "CAR", "TRUCK", "JEEP", "BIKE", "VAN", "BOAT", "TRAIN", "MOTORCYCLE", "SCOOTER", "PLANE", "MOPED"], "BEVERAGES": ["TEA", "COLA", "RUM", "WATER", "JUICE", "COFFEE", "CIDER", "WINE", "MILK", "BEER", "SODA", "LEMONADE"], "FLOWERS": ["VIOLET", "ROSE", "ORCHID", "LILY", "SUNFLOWER", "CARNATION", "DAISY", "PETUNIA", "LILAC", "MARIGOLD", "DAFFODIL", "TULIP"], "OFFICESUPPLIES": ["ERASER", "TAPE", "MARKER", "BINDER", "PAPER", "PEN", "RULER", "PENCIL", "STAPLER", "FOLDER", "GLUE", "NOTEBOOK"], "BIRDS": ["HAWK", "SPARROW", "CROW", "SEAGULL", "CARDINAL", "FINCH", "BLUEBIRD", "DOVE", "GOOSE", "PIGEON", "ROBIN", "DUCK"], "FRUIT": ["PLUM", "GRAPE", "LIME", "PEAR", "BLUEBERRY", "APPLE", "STRAWBERRY", "LEMON", "BANANA", "CHERRY", "ORANGE", "PEACH"], "INSECTS": ["BEETLE", "FLY", "BEE", "MOTH", "SPIDER", "MOSQUITO", "BUTTERFLY", "GNAT", "ANT", "ROACH", "LADYBUG", "CRICKET"], "TREES": ["MAPLE", "CEDAR", "SPRUCE", "EVERGREEN", "OAK", "HICKORY", "ELM", "PINE", "CYPRESS", "BIRCH", "WALNUT", "WILLOW"], "FARMANIMALS": ["PIG", "CALF", "COW", "TURKEY", "HEN", "ROOSTER", "LAMB", "BULL", "CHICKEN", "HORSE", "GOAT", "SHEEP"], "BODYPARTS": ["NOSE", "HEAD", "HAND", "LEG", "FOOT", "TOE", "FINGERS", "EYE", "EAR", "ELBOW", "KNEE", "ARM"], "TOYS": ["PUZZLE", "FOOTBALL", "DOLLHOUSE", "BARBIE", "FRISBEE", "BLOCKS", "BASEBALL", "BASKETBALL", "BALL", "SLINKY", "DOLL", "MARBLES"]}

# Reverse the schema to map from word to category
word_to_category = {word: category for category, words in semantic_schema.items() for word in words}

# Map each semantic field to a unique integer
category_to_int = {category: i for i, category in enumerate(semantic_schema.keys())}

# Update the word_to_category mapping to directly map words to their encoded category integers
word_to_encoded_category = {word: category_to_int[category] for word, category in word_to_category.items()}

class EEGChunkedLabeledDataset(IterableDataset):
    def __init__(self, folder_path, chunk_size=128, labels_ext='.npz'):
        super(EEGChunkedLabeledDataset, self).__init__()
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        self.labels_ext = labels_ext
        self.data_files = self._get_data_files()
        self.num_chunks = self._calculate_total_chunks()
        self.num_classes = len(semantic_schema)
        
    def _calculate_total_chunks(self):
        total_chunks = 0
        for filepath in self.data_files:
           with np.load(filepath, mmap_mode='r') as data_file:
            data = data_file['data']
            total_chunks += math.ceil(data.shape[1] / self.chunk_size)
        return total_chunks

    def __len__(self):
        return self.num_chunks

    def _get_data_files(self):
        data_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.npz')]
        return data_files

    def _get_label_file(self, data_file):
        base_name = os.path.splitext(data_file)[0]
        label_file = f"{base_name}{self.labels_ext}"
        return label_file
    
    def _read_data_and_labels(self, data_filepath):
        label_filepath = self._get_label_file(data_filepath)
        data = np.load(data_filepath)['data'] 
        labels = np.load(label_filepath)['labels']  
        
        # Map each word label to its encoded semantic category integer
        encoded_semantic_labels = np.array([word_to_encoded_category[label.upper()] for label in labels])
        
        for start_idx in range(0, data.shape[1], self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, data.shape[1])
            data_chunk = data[:, start_idx:end_idx]
            label_chunk = encoded_semantic_labels[start_idx:end_idx]
            
            yield torch.tensor(data_chunk, dtype=torch.float32), torch.tensor(label_chunk, dtype=torch.long)

    def __iter__(self):
        for data_filepath in self.data_files:
            for data_chunk, label_chunk in self._read_data_and_labels(data_filepath):
                yield data_chunk, label_chunk


