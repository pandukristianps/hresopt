from hresopt.data_loader.energy_loader import define_components, load_resources

def test_loader():
    components = define_components(
        wind=True,
        wave=True,
        battery=True
    )

    resources, meta = load_resources(
        components,
        wind_path="data/wind.nc",
        wave_path="data/wave.nc",
        lat=28,
        lon=-18
    )

    print("WIND:")
    print(resources["wind"].head())
    print(meta["wind"])

    print("\nWAVE:")
    print(resources["wave"].head())
    print(meta["wave"])


if __name__ == "__main__":
    test_loader()