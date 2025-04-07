"""Blending function models."""

import abc
from typing import Annotated, ClassVar, TypeVar

import numpy
import pydantic
from numpy.typing import ArrayLike
from pydantic_core import core_schema

from ..util import chemkin
from ..util.type_ import Frozen, Scalable, SubclassTyped

F = TypeVar("F", bound="BlendingFunction")


class BlendingFunction(Frozen, Scalable, SubclassTyped, abc.ABC):
    """Abstract base class for blending functions."""

    @abc.abstractmethod
    def __call__(self, T: ArrayLike, P_r: ArrayLike) -> numpy.ndarray:  # noqa: N803
        """Evaluate blending function, f(T, P_r).

        :param T: Temperature(s)
        :param P_r: Reduced pressure(s)
        :return: Blending function value(s)
        """
        pass


class LindemannBlendingFunction(BlendingFunction):

    # Private attributes
    type_: ClassVar[str] = "lindemann"

    def __call__(self, T: ArrayLike, P_r: ArrayLike) -> numpy.ndarray:  # noqa: N803
        """Evaluate blending function, f(T, P_r)."""
        return numpy.array(1.0)


class TroeBlendingFunction(BlendingFunction):
    A: float
    T3: float
    T1: float
    T2: float | None = None

    # Private attributes
    type_: ClassVar[str] = "troe"

    def __call__(self, T: ArrayLike, P_r: ArrayLike) -> numpy.ndarray:  # noqa: N803
        """Evaluate blending function, f(T, P_r)."""
        log_f = numpy.log10(self.f_cent(T)) / (1 + self.f1(T, P_r) ** 2)
        return numpy.power(10, log_f)

    def f_cent(self, T: ArrayLike) -> numpy.ndarray:  # noqa: N803
        """Evaluate center broadening factor."""
        f_cent = (1 - self.A) * numpy.exp(-numpy.divide(T, self.T3))
        f_cent += self.A * numpy.exp(-numpy.divide(T, self.T1))
        f_cent += 0.0 if self.T2 is None else numpy.exp(-numpy.divide(self.T2, T))
        return f_cent

    def n(self, T: ArrayLike) -> numpy.ndarray:  # noqa: N803
        """Evaluate N."""
        return 0.75 - 1.27 * numpy.log10(self.f_cent(T))

    def c(self, T: ArrayLike) -> numpy.ndarray:  # noqa: N803
        """Evaluate C."""
        return -0.4 - 0.67 * numpy.log10(self.f_cent(T))

    def f1(self, T: ArrayLike, P_r: ArrayLike) -> numpy.ndarray:  # noqa: N803
        """Evaluate f1."""
        n = self.n(T)
        c = self.c(T)
        log_p_r_plus_c = numpy.log10(P_r) + c
        return log_p_r_plus_c / (n - 0.14 * log_p_r_plus_c)


class SriBlendingFunction(BlendingFunction):
    a: float
    b: float
    c: float
    d: float = 1.0
    e: float = 0.0

    # Private attributes
    type_: ClassVar[str] = "sri"

    def __call__(self, T: ArrayLike, P_r: ArrayLike) -> numpy.ndarray:  # noqa: N803
        """Evaluate blending function, f(T, P_r)."""
        a, b, c, d, e = (self.a, self.b, self.c, self.d, self.e)
        return (
            d
            * (a * numpy.exp(-numpy.divide(b, T)) + numpy.exp(-numpy.divide(T, c)))
            ** (1 / (1 + numpy.log10(P_r) ** 2))
            * numpy.power(T, e)
        )


BlendingFunction_ = Annotated[
    pydantic.SkipValidation[BlendingFunction],
    pydantic.BeforeValidator(lambda x: BlendingFunction.model_validate(x)),
    pydantic.PlainSerializer(lambda x: BlendingFunction.model_validate(x).model_dump()),
    pydantic.GetPydanticSchema(
        lambda _, handler: core_schema.with_default_schema(handler(pydantic.BaseModel))
    ),
]


# Properties
def chemkin_aux_lines(func: BlendingFunction, head_width: int = 55) -> list[str]:
    """Determine auxiliary Chemkin lines for blending function.

    :param func: Blending function
    :return: Auxiliary lines
    """
    match func:
        case TroeBlendingFunction():
            troe_params = [func.A, func.T3, func.T1]
            troe_params += [] if func.T2 is None else [func.T2]
            return [chemkin.write_aux("TROE", troe_params, head_width=head_width)]
        case SriBlendingFunction():
            sri_params = [func.a, func.b, func.c, func.d, func.e]
            return [chemkin.write_aux("SRI", sri_params, head_width=head_width)]
        case _:
            assert isinstance(func, LindemannBlendingFunction)
            return []


# Parse helpers
def extract_blending_function_from_chemkin_parse_results(
    res: chemkin.ChemkinRateParseResults,
) -> BlendingFunction:
    """Extract blending function from Chemkin parse results.

    Chemkin parse results are modified in-place

    :param res: Chemkin parse results
    :return: Blending function
    """
    if "SRI" in res.aux_numbers:
        numbers = res.aux_numbers.pop("SRI")
        keys = ["a", "b", "c", "d", "e"]
        data = dict(zip(keys, numbers, strict=False))
        return SriBlendingFunction.model_validate(data)

    if "TROE" in res.aux_numbers:
        numbers = res.aux_numbers.pop("TROE")
        keys = ["A", "T3", "T1", "T2"]
        data = dict(zip(keys, numbers, strict=False))
        return TroeBlendingFunction.model_validate(data)

    return LindemannBlendingFunction()
