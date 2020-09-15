try:
    from krate.fileutils import save, load
except:
    import traceback
    print("Failed to find krate: install from:")
    print("git@github.com:BenedictWilkinsAI/krate.git")
    traceback.print_exc()
