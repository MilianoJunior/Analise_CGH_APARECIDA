from pathlib import Path
import os

def check_file_sizes(directory='.'):
    """
    Verifica o tamanho dos arquivos em um diretório.
    
    Args:
        directory (str): Caminho do diretório a ser verificado
        
    Returns:
        dict: Dicionário com nome do arquivo e seu tamanho
    """
    def format_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    file_sizes = {}
    dir_path = Path(directory)
    
    if dir_path.exists():
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                size = file_path.stat().st_size
                file_sizes[str(file_path.relative_to(dir_path))] = format_size(size)
    
    return file_sizes

# Exemplo de uso
if __name__ == '__main__':
    sizes = check_file_sizes('data')
    print("\nTamanho dos arquivos:")
    for file, size in sizes.items():
        print(f"- {file}: {size}")