import torch
from torch.utils.data import Dataset, DataLoader
import json
from src.config import DATA_DIR, BATCH_SIZE
from src.dataset import load_split

class ShakespeareDataset(Dataset):
    """
    Clase personalizada de PyTorch para leer tus datos de Shakespeare.
    Convierte las cadenas de 80 caracteres en listas de 80 números enteros.
    """
    def __init__(self, split_name):
        # 1. Cargamos el DataFrame (train, val o test) usando tu función existente
        self.df = load_split(split_name)
        
        # 2. Cargamos el diccionario de vocabulario para traducir letras a números
        vocab_path = DATA_DIR / "vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            
        self.stoi = self.vocab["stoi"]          # Diccionario: Letra -> Número
        self.unk_idx = self.vocab["unk_index"]  # Índice para caracteres desconocidos (0)

    def __len__(self):
        # Le dice a PyTorch cuántas filas tiene este split
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Obtenemos la fila actual
        row = self.df.iloc[idx]
        
        # 2. Extraemos el texto de 80 caracteres (x) y el target numérico (y_id)
        text_x = row["x"]
        y_target = row["y_id"]
        
        # 3. Convertimos cada una de las 80 letras en su número correspondiente
        # Si una letra no existe en el vocabulario, usa el token <UNK> (0)
        x_encoded = [self.stoi.get(char, self.unk_idx) for char in text_x]
        
        # 4. Retornamos los tensores de PyTorch (tipo 'long' para índices enteros)
        return torch.tensor(x_encoded, dtype=torch.long), torch.tensor(y_target, dtype=torch.long)

# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    print("Preparando los DataLoaders...")
    
    # Instanciamos los datasets
    train_dataset = ShakespeareDataset("train")
    val_dataset = ShakespeareDataset("val")
    
    # DataLoader agrupa las filas individuales en lotes del tamaño de BATCH_SIZE (128)
    # shuffle=True mezcla los datos de entrenamiento para que el modelo no memorice el orden
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Comprobación de dimensiones
    x_batch, y_batch = next(iter(train_loader))
    print(f"Forma del tensor X (Entrada): {x_batch.shape} -> Debería ser [{BATCH_SIZE}, 80]")
    print(f"Forma del tensor Y (Target):  {y_batch.shape} -> Debería ser [{BATCH_SIZE}]")
