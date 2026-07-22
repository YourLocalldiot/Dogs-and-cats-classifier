from pathlib import Path


def ensure_project_structure():
    """Create required directories if they are missing."""
    base_dir = Path(__file__).resolve().parent.parent
    required_dirs = [
        base_dir / "PetImages" / "Cat",
        base_dir / "PetImages" / "Dog",
        base_dir / "models",
    ]

    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
