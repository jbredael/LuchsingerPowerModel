"""Core power estimation model for airborne wind energy systems based on the Luchsinger model [1].

This module provides the main PowerModel class for calculating power output
of pumping kite systems. The model is planet-agnostic and configurable via
dictionaries or YAML files. Supports both legacy format and awesIO format.

References:
    [1] R.H. Luchsinger: "Pumping cycle kite power". Springer, 2013.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import copy
import logging

import numpy as np
import yaml
from scipy import optimize as op

from awesio.validator import validate as awesio_validate

from src.power_luchsinger.calculations import (
    calculate_force_factor_out,
    calculate_force_factor_in,
    calculate_tether_force_out,
    calculate_tether_force_in,
    calculate_cycle_results
)

# Configure logger
logger = logging.getLogger(__name__)


class PowerModel:
    """Calculate power output for airborne wind energy systems.

    This model is configurable via YAML files or
    dictionaries. All physical parameters must be provided through
    configuration.

    The model implements the Luchsinger pumping cycle model [1].
    """


    def __init__(self, config: Dict[str, Any], simulation_settings: Dict[str, Any] = None):
        """Initialize power model with configuration parameters.

        Args:
            config: Dictionary containing model parameters (legacy format)
                or awesIO system configuration.
            simulation_settings: Optional simulation settings dictionary
                for awesIO format configs.

        Raises:
            ValueError: If required configuration keys are missing.
            ValueError: If parameter values are physically invalid.
        """
        self.config = config
        self.simulation_settings = simulation_settings

        # Extract parameters (handles both legacy and awesIO formats)
        self._extract_parameters()

        # Then validate physical constraints
        self._validate_physical_constraints()

        # Finally compute derived parameters
        self._compute_derived_parameters()

    def _validate_physical_constraints(self) -> None:
        """Validate that parameter values are physically reasonable.

        Raises:
            ValueError: If values are physically invalid.
        """
        if self.wingArea <= 0:
            raise ValueError("Wing area must be positive")
        if self.liftCoefficientOut <= 0:
            raise ValueError("Lift coefficient must be positive")
        if self.dragCoefficientKiteOut <= 0:
            raise ValueError("Drag coefficient must be positive")

        if self.tetherMaxLength <= self.tetherMinLength:
            raise ValueError("Max tether length must exceed min length")

        if self.airDensity <= 0:
            raise ValueError("Air density must be positive")

        if self.nominalTetherForce <= 0:
            raise ValueError("Nominal tether force must be positive")
        if self.nominalGeneratorPower <= 0:
            raise ValueError("Nominal generator power must be positive")

        if self.cutInWindSpeed >= self.cutOutWindSpeed:
            raise ValueError("Cut-in wind speed must be less than cut-out")

    def _extract_parameters(self) -> None:
        """Extract parameters directly from awesIO format config."""
        components = self.config.get('components', {})

        # Extract operational and atmosphere parameters from simulation settings
        operational = self.simulation_settings.get('operational', {})
        atmosphere = self.simulation_settings.get('atmosphere', {})
        
        # Extract operational parameters
        self.cutInWindSpeed = operational.get('cut_in_wind_speed_m_s')
        self.cutOutWindSpeed = operational.get('cut_out_wind_speed_m_s')
        self.elevationAngleOut = np.radians(operational.get('elevation_angle_out_deg'))
        self.elevationAngleIn = np.radians(operational.get('elevation_angle_in_deg'))
        self.reelOutSpeedLimit = operational.get('max_reel_out_speed_m_s')
        self.reelInSpeedLimit = operational.get('max_reel_in_speed_m_s')
        self.tetherMinLength = operational.get('minimum_tether_length_m')

        # Extract atmosphere parameters
        self.airDensity = atmosphere.get('air_density_kg_m3')

        # Extract wing parameters
        wing = components.get('wing', {})
        wing_structure = wing.get('structure', {})
        wing_aero = wing.get('aerodynamics', {}).get('simple_aero_model', {})
        
        self.wingArea = wing_structure.get('projected_surface_area_m2')
        self.liftCoefficientOut = wing_aero.get('lift_coefficient_reel_out')
        self.dragCoefficientKiteOut = wing_aero.get('drag_coefficient_reel_out')
        self.dragCoefficientKiteIn = wing_aero.get('drag_coefficient_reel_in')

        # Extract tether parameters
        tether = components.get('tether', {})
        tether_structure = tether.get('structure', {})       
        self.tetherMaxLength = tether_structure.get('length_m')


        # Extract ground station parameters
        ground_station = components.get('ground_station', {})
        drum = ground_station.get('drum', {})
        generator = ground_station.get('generator', {})
        storage = ground_station.get('storage', {})
        
        self.nominalTetherForce = drum.get('max_tether_force_n') or tether_structure.get('max_tether_force_n')
        self.nominalGeneratorPower = generator.get('rated_power_kw', 0) * 1000  # kW to W
        self.generatorEfficiency = generator.get('efficiency')
        self.storageEfficiency = storage.get('efficiency')


    def _compute_derived_parameters(self) -> None:
        """Compute derived parameters from base configuration."""
        # Operational tether length
        self.operationalLength = (
            (self.tetherMaxLength - self.tetherMinLength) / 2 +
            self.tetherMinLength
        )
        self.reelingLength = self.tetherMaxLength - self.tetherMinLength

        # Total drag coefficients
        self.dragCoefficientOut = self.dragCoefficientKiteOut
        self.dragCoefficientIn = self.dragCoefficientKiteIn

        # Force factors
        self.forceFactorOut = calculate_force_factor_out(
            self.liftCoefficientOut, self.dragCoefficientOut
        )
        self.forceFactorIn = calculate_force_factor_in(self.dragCoefficientIn)

        # Compute nominal wind speeds for force and power limits
        self._compute_nominal_wind_speeds()

    def _compute_nominal_wind_speeds(self) -> None:
        """Compute wind speeds at which force and power limits are reached."""
        windSpeeds = np.arange(self.cutInWindSpeed, self.cutOutWindSpeed, 0.1)

        self.nominalWindSpeedForce = self.cutOutWindSpeed
        self.nominalGammaOutForce = 0.33

        self.nominalWindSpeedPower = self.cutOutWindSpeed
        self.nominalGammaOutPower = 0.33

        # Find force limit
        for vw in windSpeeds:
            gammaOutMax = self.reelOutSpeedLimit / vw
            gammaInMax = self.reelInSpeedLimit / vw

            gammaOut, gammaIn = self._optimize_gamma_out_in_region1(
                self.elevationAngleOut, self.elevationAngleIn,
                self.forceFactorOut, self.forceFactorIn,
                gammaOutMax, gammaInMax
            )

            tetherForce = calculate_tether_force_out(
                self.airDensity, vw, self.wingArea,
                gammaOut, self.elevationAngleOut, self.forceFactorOut
            )

            if tetherForce >= self.nominalTetherForce:
                self.nominalWindSpeedForce = vw
                self.nominalGammaOutForce = gammaOut
                break

        # Find power limit (only for winds above force limit)
        for vw in windSpeeds:
            if vw <= self.nominalWindSpeedForce:
                continue

            mu = vw / self.nominalWindSpeedForce
            gammaOut = (
                np.cos(self.elevationAngleOut) -
                (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce) / mu
            )
            vOut = vw * gammaOut

            # Simple power calculation
            mechPower = self.nominalTetherForce * vOut
            elecPower = mechPower * self.generatorEfficiency

            if elecPower >= self.nominalGeneratorPower:
                self.nominalWindSpeedPower = vw
                self.nominalGammaOutPower = gammaOut
                break

        # Compute nominal reel-out speed for power-limited region
        self.nominalReelOutSpeed = (
            self.nominalGeneratorPower / 
            (self.nominalTetherForce * self.generatorEfficiency)
        )


    def calculate_power(self,
                       windSpeed: float,
                       airDensity: float = None) -> Dict[str, float]:
        """Calculate power output for given wind speed.

        Args:
            windSpeed (float): Wind speed in m/s.
            airDensity (float): Air density in kg/m³. If None, uses
                atmosphere.airDensity from config.

        Returns:
            Dict[str, float]: Dictionary with keys:
                - 'cyclePower': Average cycle power (W)
                - 'reelOutPower': Reel-out power (W)
                - 'reelInPower': Reel-in power (W)
                - 'reelOutTime': Reel-out time (s)
                - 'reelInTime': Reel-in time (s)
                - 'tetherForceOut': Tether force during reel-out (N)
                - 'tetherForceIn': Tether force during reel-in (N)
                - 'reelOutSpeed': Reel-out speed (m/s)
                - 'reelInSpeed': Reel-in speed (m/s)
                - 'gammaOut': Reel-out factor (-)
                - 'gammaIn': Reel-in factor (-)
        """
        if airDensity is None:
            airDensity = self.airDensity

        if windSpeed < self.cutInWindSpeed or windSpeed > self.cutOutWindSpeed:
            return {
                'cyclePower': 0.0,
                'reelOutPower': 0.0,
                'reelInPower': 0.0,
                'reelOutTime': 0.0,
                'reelInTime': 0.0,
                'tetherForceOut': 0.0,
                'tetherForceIn': 0.0,
                'reelOutSpeed': 0.0,
                'reelInSpeed': 0.0,
                'gammaOut': 0.0,
                'gammaIn': 0.0,
            }

        if windSpeed < self.nominalWindSpeedForce:
            return self._calculate_power_region1(windSpeed, airDensity)
        elif windSpeed < self.nominalWindSpeedPower:
            return self._calculate_power_region2(windSpeed, airDensity)
        else:
            return self._calculate_power_region3(windSpeed, airDensity)

    def _calculate_power_region1(self,
                                  windSpeed: float,
                                  airDensity: float) -> Dict[str, float]:
        """Calculate power in Region 1 (below force limit).

        Args:
            windSpeed (float): Wind speed in m/s.
            airDensity (float): Air density in kg/m³.

        Returns:
            Dict with power and time details.
        """
        gammaOutMax = self.reelOutSpeedLimit / windSpeed
        gammaInMax = self.reelInSpeedLimit / windSpeed

        gammaOut, gammaIn = self._optimize_gamma_out_in_region1(
            self.elevationAngleOut, self.elevationAngleIn,
            self.forceFactorOut, self.forceFactorIn,
            gammaOutMax, gammaInMax
        )

        vOut = windSpeed * gammaOut
        vIn = windSpeed * gammaIn

        tetherForceOut = calculate_tether_force_out(
            airDensity, windSpeed, self.wingArea,
            gammaOut, self.elevationAngleOut, self.forceFactorOut
        )
        tetherForceIn = calculate_tether_force_in(
            airDensity, windSpeed, self.wingArea,
            gammaIn, self.elevationAngleIn, self.forceFactorIn
        )

        return calculate_cycle_results(
            tetherForceOut, tetherForceIn, vOut, vIn,
            self.reelingLength, gammaOut, gammaIn,
            self.generatorEfficiency, self.storageEfficiency
        )

    def _optimize_gamma_out_in_region1(self,
                                        elevationAngleOut: float,
                                        elevationAngleIn: float,
                                        forceFactorOut: float,
                                        forceFactorIn: float,
                                        gammaOutMax: float,
                                        gammaInMax: float) -> Tuple[float, float]:
        """Calculate optimal dimensionless reeling velocity factors.
        
        Optimizes the cycle power factor by finding the optimal reeling
        velocities for both reel-out and reel-in phases.
        
        Args:
            elevationAngleOut (float): Elevation angle during reel-out in rad.
            elevationAngleIn (float): Elevation angle during reel-in in rad.
            forceFactorOut (float): Force factor during reel-out.
            forceFactorIn (float): Force factor during reel-in.
            gammaOutMax (float): Maximum gamma_out (v_out_max / v_wind).
            gammaInMax (float): Maximum gamma_in (v_in_max / v_wind).
            
        Returns:
            Tuple[float, float]: (optimal gamma_out, optimal gamma_in).
        """
        from scipy import optimize as op
        
        def objective(x):
            gammaOut, gammaIn = x
            # Cycle power factor from Luchsinger model
            powerFactor = (
                (np.cos(elevationAngleOut) - gammaOut)**2 -
                (forceFactorIn / forceFactorOut) *
                (1 + 2 * np.cos(elevationAngleIn) * gammaIn + gammaIn**2)
            ) * ((gammaOut * gammaIn) / (gammaOut + gammaIn))
            return -powerFactor  # Minimize negative = maximize
        
        bounds = ((0.001, gammaOutMax), (0.001, gammaInMax))
        result = op.minimize(objective, (0.001, 0.001), bounds=bounds, method='SLSQP')
        
        return result['x'][0], result['x'][1]

    def _calculate_power_region2(self,
                                  windSpeed: float,
                                  airDensity: float) -> Dict[str, float]:
        """Calculate power in Region 2 (force-limited, below power limit).

        Args:
            windSpeed (float): Wind speed in m/s.
            airDensity (float): Air density in kg/m³.

        Returns:
            Dict with power and time details.
        """
        mu = windSpeed / self.nominalWindSpeedForce
        gammaInMax = self.reelInSpeedLimit / windSpeed

        gammaOut = (
            np.cos(self.elevationAngleOut) -
            (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce) / mu
        )
        vOut = windSpeed * gammaOut
        tetherForceOut = self.nominalTetherForce

        gammaIn = self._optimize_gamma_in_region2(mu, gammaInMax)
        vIn = windSpeed * gammaIn

        tetherForceIn = calculate_tether_force_in(
            airDensity, windSpeed, self.wingArea,
            gammaIn, self.elevationAngleIn, self.forceFactorIn
        )

        return calculate_cycle_results(
            tetherForceOut, tetherForceIn, vOut, vIn,
            self.reelingLength, gammaOut, gammaIn,
            self.generatorEfficiency, self.storageEfficiency
        )

    def _optimize_gamma_in_region2(self, mu: float, gammaInMax: float) -> float:
        """Optimize gamma_in for Region 2 operation.

        Args:
            mu (float): Wind speed ratio to nominal force wind speed.
            gammaInMax (float): Maximum gamma_in.

        Returns:
            float: Optimal gamma_in.
        """
        def objective(x):
            gammaIn = x[0]
            gammaOutEff = (
                mu * np.cos(self.elevationAngleOut) -
                np.cos(self.elevationAngleOut) +
                self.nominalGammaOutForce
            )

            powerFactor = (
                (1 / mu**2) * (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce)**2 -
                (self.forceFactorIn / self.forceFactorOut) *
                (1 + 2 * np.cos(self.elevationAngleIn) * gammaIn + gammaIn**2)
            ) * (
                (gammaIn * gammaOutEff) /
                (mu * gammaIn + gammaOutEff)
            )
            return -powerFactor

        result = op.minimize(
            objective, [0.001],
            bounds=[(0.001, gammaInMax)],
            method='SLSQP'
        )

        return result['x'][0]

    def _calculate_power_region3(self,
                                  windSpeed: float,
                                  airDensity: float) -> Dict[str, float]:
        """Calculate power in Region 3 (power-limited).

        Args:
            windSpeed (float): Wind speed in m/s.
            airDensity (float): Air density in kg/m³.

        Returns:
            Dict with power and time details.
        """
        mu = windSpeed / self.nominalWindSpeedPower
        gammaInMax = self.reelInSpeedLimit / windSpeed

        vOut = self.nominalReelOutSpeed
        gammaOut = vOut / windSpeed
        elecPowerOut = self.nominalGeneratorPower
        tetherForceOut = self.nominalTetherForce

        gammaIn = self._optimize_gamma_in_region3(mu, gammaInMax)
        vIn = windSpeed * gammaIn

        tetherForceIn = calculate_tether_force_in(
            airDensity, windSpeed, self.wingArea,
            gammaIn, self.elevationAngleIn, self.forceFactorIn
        )

        return calculate_cycle_results(
            tetherForceOut, tetherForceIn, vOut, vIn,
            self.reelingLength, gammaOut, gammaIn,
            self.generatorEfficiency, self.storageEfficiency
        )

    def _optimize_gamma_in_region3(self, mu: float, gammaInMax: float) -> float:
        """Optimize gamma_in for Region 3 operation.

        Args:
            mu (float): Wind speed ratio to nominal power wind speed.
            gammaInMax (float): Maximum gamma_in.

        Returns:
            float: Optimal gamma_in.
        """
        def objective(x):
            gammaIn = x[0]

            powerFactor = (
                (1 / mu**2) *
                (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce)**2 -
                (self.forceFactorIn / self.forceFactorOut) *
                (1 + 2 * np.cos(self.elevationAngleIn) * gammaIn + gammaIn**2)
            ) * (
                (self.nominalGammaOutPower * gammaIn) /
                (self.nominalGammaOutPower + mu * gammaIn)
            )
            return -powerFactor

        result = op.minimize(
            objective, [0.001],
            bounds=[(0.001, gammaInMax)],
            method='SLSQP'
        )

        return result['x'][0]

    def generate_power_curve(self,
                            numPoints: int = 100) -> Dict[str, np.ndarray]:
        """Generate power curve from cut-in to cut-out wind speed.

        Args:
            numPoints (int): Number of wind speed points. Defaults to 100.

        Returns:
            Dict[str, np.ndarray]: Dictionary with keys:
                - 'windSpeed': Wind speed array in m/s
                - 'power': Cycle power output array in W
                - 'reelOutPower': Reel-out power array in W
                - 'reelInPower': Reel-in power array in W
                - 'reelOutTime': Reel-out time array in s
                - 'reelInTime': Reel-in time array in s
                - 'tetherForceOut': Tether force during reel-out array in N
                - 'tetherForceIn': Tether force during reel-in array in N
                - 'reelOutSpeed': Reel-out speed array in m/s
                - 'reelInSpeed': Reel-in speed array in m/s
                - 'gammaOut': Reel-out factor array (-)
                - 'gammaIn': Reel-in factor array (-)
        """
        windSpeeds = np.linspace(self.cutInWindSpeed, self.cutOutWindSpeed, numPoints)
        
        results = [self.calculate_power(ws) for ws in windSpeeds]

        return {
            'windSpeed': windSpeeds,
            'power': np.array([r['cyclePower'] for r in results]),
            'reelOutPower': np.array([r['reelOutPower'] for r in results]),
            'reelInPower': np.array([r['reelInPower'] for r in results]),
            'reelOutTime': np.array([r['reelOutTime'] for r in results]),
            'reelInTime': np.array([r['reelInTime'] for r in results]),
            'tetherForceOut': np.array([r['tetherForceOut'] for r in results]),
            'tetherForceIn': np.array([r['tetherForceIn'] for r in results]),
            'reelOutSpeed': np.array([r['reelOutSpeed'] for r in results]),
            'reelInSpeed': np.array([r['reelInSpeed'] for r in results]),
            'gammaOut': np.array([r['gammaOut'] for r in results]),
            'gammaIn': np.array([r['gammaIn'] for r in results]),
        }

    @classmethod
    def from_yaml(
        cls,
        yamlPath: Path,
        simulationSettingsPath: Path = None,
        validate: bool = True
    ) -> 'PowerModel':
        """Load configuration from YAML file and create model instance.

        Supports both legacy format and awesIO format configuration files.
        If the file is in awesIO format, it will be validated before use.

        Args:
            yamlPath: Path to system YAML configuration file.
            simulationSettingsPath: Path to simulation settings YAML file
                containing operational and atmosphere parameters.
                Required for awesIO format system configs.
            validate: Whether to validate awesIO format files. Defaults to True.

        Returns:
            PowerModel: Initialized power model instance.

        Raises:
            FileNotFoundError: If YAML file doesn't exist.
            ValueError: If YAML contains invalid configuration.
            Exception: If awesIO validation fails.
        """
        yamlPath = Path(yamlPath)

        if not yamlPath.exists():
            raise FileNotFoundError(f"Configuration file not found: {yamlPath}")

        with open(yamlPath, 'r') as f:
            config = yaml.safe_load(f)

        # Load simulation settings if provided
        simulation_settings = None
        if simulationSettingsPath is not None:
            simulationSettingsPath = Path(simulationSettingsPath)
            if not simulationSettingsPath.exists():
                raise FileNotFoundError(
                    f"Simulation settings file not found: {simulationSettingsPath}"
                )
            with open(simulationSettingsPath, 'r') as f:
                simulation_settings = yaml.safe_load(f)
            print(f"Loaded simulation settings from: {simulationSettingsPath.name}")

        # Validate using awesIO validator (if schema exists)
        if validate:
            try:
                awesio_validate(
                    input=yamlPath,
                    restrictive=False,
                    defaults=False,
                )
                print(f"  ✓ {yamlPath.name} validated against system_schema")
            except FileNotFoundError:
                print(f"  Note: system_schema not available, skipping validation")

        # Create model with awesIO config and simulation settings
        return cls(config, simulation_settings)

    def export_power_curves_awesio(
        self,
        data: Dict[str, np.ndarray],
        output_path: Path,
        name: str = "Luchsinger Power Curves",
        description: str = "Power curves for pumping ground-gen AWE system",
        note: str = "Power curve data generated from Luchsinger model",
        validate: bool = True,
    ) -> None:
        """Export power curve data in awesIO format.

        Args:
            data: Power curve data dictionary from generate_power_curve().
            output_path: Path to save the output YAML file.
            name: Name for the power curves dataset.
            description: Description of the power curves.
            note: Additional notes about the data.
            validate: Whether to validate the output file. Defaults to True.
        """
        output_path = Path(output_path)

        # Calculate operating altitude from tether length and elevation angle
        operating_altitude = self.operationalLength * np.sin(self.elevationAngleOut)

        # Number of wind speed points
        numPoints = len(data['windSpeed'])

        # Build awesIO format output
        output = {
            'metadata': {
                'name': name,
                'description': description,
                'note': note,
                'awesIO_version': '0.1.0',
                'schema': 'power_curves_schema.yml',
                'time_created': datetime.now().isoformat(),
                'model_config': {
                    'wing_area_m2': float(self.wingArea),
                    'nominal_power_w': float(self.nominalGeneratorPower),
                    'nominal_tether_force_n': float(self.nominalTetherForce),
                    'cut_in_wind_speed_m_s': float(self.cutInWindSpeed),
                    'cut_out_wind_speed_m_s': float(self.cutOutWindSpeed),
                    'operating_altitude_m': float(operating_altitude),
                    'tether_length_operational_m': float(self.operationalLength),
                },
            },
            'altitudes_m': [float(operating_altitude)],  # Single altitude for this model
            'reference_wind_speeds_m_s': [float(v) for v in data['windSpeed']],
            'power_curves': [
                {
                    'profile_id': 1,
                    'speed_ratio_at_operating_altitude': 1.0,
                    'u_normalized': [1.0],  # Single altitude, normalized to 1
                    'v_normalized': [0.0],  # No crosswind component
                    'probability_weight': 1.0,
                    'cycle_power_w': [float(p) for p in data['power']],
                    'reel_out_power_w': [float(p) for p in data['reelOutPower']],
                    'reel_in_power_w': [float(p) for p in data['reelInPower']],
                    'reel_out_time_s': [float(t) for t in data['reelOutTime']],
                    'reel_in_time_s': [float(t) for t in data['reelInTime']],
                    'cycle_time_s': [
                        float(t_out + t_in)
                        for t_out, t_in in zip(data['reelOutTime'], data['reelInTime'])
                    ],
                },
            ],
        }

        # Write output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

        # Validate output if requested
        if validate:
            awesio_validate(
                input=output_path,
                restrictive=False,
                defaults=False,
            )
            print(f"  ✓ Output validated against power_curves_schema")
