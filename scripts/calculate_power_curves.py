"""Calculate and plot comprehensive power curve analysis for an AWE system.

This script loads configuration from an awesIO YAML file, validates it,
calculates the power curve from cut-in to cut-out wind speed, and creates
a comprehensive visualization. Results are exported in awesIO format.
"""

import sys
from pathlib import Path
import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.power_luchsinger import PowerModel
from src.power_luchsinger.plotting import plot_comprehensive_analysis, extract_model_params


def print_summary(model: PowerModel, data: dict) -> None:
    """Print summary of the power curve calculation.

    Args:
        model: The power model instance.
        data: Dictionary with all power curve data arrays.
    """
    power = data['power']
    windSpeed = data['windSpeed']

    print("\n" + "=" * 60)
    print("POWER CURVE SUMMARY")
    print("=" * 60)
    print(f"\nSystem Parameters:")
    print(f"  Wing Area:              {model.wingArea:.1f} m²")
    print(f"  Air Density:            {model.airDensity:.3f} kg/m³")
    print(f"  Nominal Tether Force:   {model.nominalTetherForce:.0f} N")
    print(f"  Nominal Generator Power:{model.nominalGeneratorPower/1000:.1f} kW")
    print(f"  Tether Length:          {model.tetherMinLength:.0f} - {model.tetherMaxLength:.0f} m")

    print(f"\nOperational Envelope:")
    print(f"  Cut-in Wind Speed:      {model.cutInWindSpeed:.1f} m/s")
    print(f"  Cut-out Wind Speed:     {model.cutOutWindSpeed:.1f} m/s")
    print(f"  Force Limit Wind Speed: {model.nominalWindSpeedForce:.1f} m/s")
    print(f"  Power Limit Wind Speed: {model.nominalWindSpeedPower:.1f} m/s")

    print(f"\nPower Curve Statistics:")
    print(f"  Maximum Power:          {np.max(power)/1000:.2f} kW")
    print(f"  Wind Speed at Max Power:{windSpeed[np.argmax(power)]:.1f} m/s")
    print("=" * 60)


def main():
    """Main entry point for power curve calculation script."""
    # Use awesIO format configuration files
    systemConfigPath = workspace_root / 'data' / 'soft_kite_pumping_ground_gen_system.yml'
    simulationSettingsPath = workspace_root / 'data' / 'simulation_settings_config.yml'

    if not systemConfigPath.exists():
        print(f"Error: System config file not found: {systemConfigPath}")
        sys.exit(1)

    if not simulationSettingsPath.exists():
        print(f"Error: Simulation settings file not found: {simulationSettingsPath}")
        sys.exit(1)

    print(f"Loading system configuration from: {systemConfigPath}")
    print(f"Loading simulation settings from: {simulationSettingsPath}")

    # Load power model with awesIO validation
    model = PowerModel.from_yaml(
        systemConfigPath,
        simulationSettingsPath=simulationSettingsPath,
        validate=True
    )

    # Generate power curve with 500 points
    print(f"\nCalculating power curve (500 points)...")
    data = model.generate_power_curve(numPoints=500)

    # Print summary
    print_summary(model, data)

    # Export power curves in awesIO format
    output_path = workspace_root / 'results' / 'luchsinger_power_curves.yml'
    print(f"\nExporting power curves to: {output_path}")
    model.export_power_curves_awesio(
        data=data,
        output_path=output_path,
        name="Luchsinger Model Power Curves",
        description="Power curves for 100kW soft kite pumping ground-gen AWE system",
        note="Generated using Luchsinger pumping cycle model",
    )

    # Extract model parameters
    model_params = extract_model_params(model)

    # Create comprehensive plot with energy subplot
    print("\nGenerating comprehensive analysis plots...")
    plot_comprehensive_analysis(
        data,
        model_params,
        save_path="results/power_curve_analysis.png",
        show=False
    )
    print("Plots saved to: results/power_curve_analysis.png")


if __name__ == '__main__':
    main()
