from typing import List, Union, Dict, Tuple, Callable, Optional

VideoAnnotation = Optional[Dict[str, Union[str, List[Dict[str, str]]]]]
Metadata = List[Tuple[str, str]]