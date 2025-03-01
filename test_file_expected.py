from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from pandas import DataFrame
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.scales import Scale


class Mappable:
    def __init__(
        self,
        val: Any = None,
        depend: str | None = None,
        rc: str | None = None,
        auto: bool = False,
        grouping: bool = True,
    ):
        """
        Property that can be mapped from data or set directly, with flexible defaults.

        Parameters
        ----------
        val : Any
            Use this value as the default.
        depend : str
            Use the value of this feature as the default.
        rc : str
            Use the value of this rcParam as the default.
        auto : bool
            The default value will depend on other parameters at compile time.
        grouping : bool
            If True, use the mapped variable to define groups.

        """
        if depend is not None:
            assert depend in PROPERTIES
        if rc is not None:
            assert rc in mpl.rcParams

        self._val = val
        self._rc = rc
        self._depend = depend
        self._auto = auto
        self._grouping = grouping

    def __repr__(self):
        """Nice formatting for when object appears in Mark init signature."""
        if self._val is not None:
            s = f"<{repr(self._val)}>"
        elif self._depend is not None:
            s = f"<depend:{self._depend}>"
        elif self._rc is not None:
            s = f"<rc:{self._rc}>"
        elif self._auto:
            s = "<auto>"
        else:
            s = "<undefined>"
        return s

    @property
    def depend(self) -> Any:
        """Return the name of the feature to source a default value from."""
        return self._depend

    @property
    def grouping(self) -> bool:
        return self._grouping

    @property
    def default(self) -> Any:
        """Get the default value for this feature, or access the relevant rcParam."""
        if self._val is not None:
            return self._val
        elif self._rc is not None:
            return mpl.rcParams.get(self._rc)
        

@dataclass
class Mark:
    """Base class for objects that visually represent data."""

    artist_kws: dict = field(default_factory=dict)

    @property
    def _mappable_props(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if isinstance(f.default, Mappable)}

    @property
    def _grouping_props(self):
        # TODO does it make sense to have variation within a Mark's
        # properties about whether they are grouping?
        return [f.name for f in fields(self) if isinstance(f.default, Mappable) and f.default.grouping]

    # TODO make this method private? Would extender every need to call directly?
    def _resolve(self, data: DataFrame | dict[str, Any], name: str, scales: dict[str, Scale] | None = None) -> Any:
        """Obtain default, specified, or mapped value for a named feature.

        Parameters
        ----------
        data : DataFrame or dict with scalar values
            Container with data values for features that will be semantically mapped.
        name : string
            Identity of the feature / semantic.
        scales: dict
            Mapping from variable to corresponding scale object.

        Returns
        -------
        value or array of values
            Outer return type depends on whether `data` is a dict (implying that
            we want a single value) or DataFrame (implying that we want an array
            of values with matching length).

        """
        feature = self._mappable_props[name]
        prop = PROPERTIES.get(name, Property(name))
        directly_specified = not isinstance(feature, Mappable)
        return_multiple = isinstance(data, pd.DataFrame)
        return_array = return_multiple and not name.endswith("style")

        # Special case width because it needs to be resolved and added to the dataframe
        # during layer prep (so the Move operations use it properly).
        # TODO how does width *scaling* work, e.g. for violin width by count?
        if name == "width":
            directly_specified = directly_specified and name not in data

        if directly_specified:
            feature = prop.standardize(feature)
            if return_multiple:
                feature = [feature] * len(data)
            if return_array:
                feature = np.array(feature)
            return feature

        if name in data:
            if scales is None or name not in scales:
                # TODO Might this obviate the identity scale? Just don't add a scale?
                feature = data[name]
            else:
                scale = scales[name]
                value = data[name]
                try:
                    feature = scale(value)
                except Exception as err:
                    raise PlotSpecError._during("Scaling operation", name) from err

            if return_array:
                feature = np.asarray(feature)
            return feature

        if feature.depend is not None:
            # TODO add source_func or similar to transform the source value?
            # e.g. set linewidth as a proportion of pointsize?
            return self._resolve(data, feature.depend, scales)

        default = prop.standardize(feature.default)
        if return_multiple:
            default = [default] * len(data)
        if return_array:
            default = np.array(default)
        return default

    def _infer_orient(self, scales: dict) -> str:  # TODO type scales
        # TODO The original version of this (in seaborn._base) did more checking.
        # Paring that down here for the prototype to see what restrictions make sense.

        # TODO rethink this to map from scale type to "DV priority" and use that?
        # e.g. Nominal > Discrete > Continuous

        x = 0 if "x" not in scales else scales["x"]._priority
        y = 0 if "y" not in scales else scales["y"]._priority

        if y > x:
            return "y"
        else:
            return "x"

    def _plot(self, split_generator: Callable[[], Generator], scales: dict[str, Scale], orient: str) -> None:
        """Main interface for creating a plot."""
        raise NotImplementedError()

    def _legend_artist(self, variables: list[str], value: Any, scales: dict[str, Scale]) -> Artist | None:
        return None


def resolve_properties(mark: Mark, data: DataFrame, scales: dict[str, Scale]) -> dict[str, Any]:
    props = {name: mark._resolve(data, name, scales) for name in mark._mappable_props}
    return props
