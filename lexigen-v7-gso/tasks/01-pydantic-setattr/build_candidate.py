from __future__ import annotations

import argparse
from pathlib import Path

START = "        def __setattr__(self, name: str, value: Any) -> None:\n"
END = "        def __delattr__(self, item: str) -> Any:\n"
CLASSVAR_ANCHOR = "    __pydantic_fields__: ClassVar[Dict[str, FieldInfo]]  # noqa: UP006\n"
CLASSVAR_INSERT = "    __pydantic_setattr_handlers__: ClassVar[Dict[str, Callable[[BaseModel, str, Any], None]]] = {}\n\n"
META_ANCHOR = "            cls = cast('type[BaseModel]', super().__new__(mcs, cls_name, bases, namespace, **kwargs))\n"

COMMON_HEAD = '''        def __setattr__(self, name: str, value: Any) -> None:
            cls = self.__class__
            if name in cls.__class_vars__:
                raise AttributeError(
                    f'{name!r} is a ClassVar of `{cls.__name__}` and cannot be set on an instance. '
                    f'If you want to set a value on the class, use `{cls.__name__}.{name} = value`.'
                )
            elif not _fields.is_valid_field_name(name):
                private_attributes = cls.__private_attributes__
                if self.__pydantic_private__ is None or name not in private_attributes:
                    _object_setattr(self, name, value)
                else:
                    attribute = private_attributes[name]
                    if hasattr(attribute, '__set__'):
                        attribute.__set__(self, value)  # type: ignore
                    else:
                        self.__pydantic_private__[name] = value
                return

            self._check_frozen(name, value)
'''

TAIL = '''            attr = getattr(cls, name, None)
            if isinstance(attr, property):
                attr.__set__(self, value)
            elif isinstance(attr, cached_property):
                self.__dict__[name] = value
            elif config.get('validate_assignment', None):
                self.__pydantic_validator__.validate_assignment(self, name, value)
            elif config.get('extra') != 'allow' and name not in fields:
                raise ValueError(f'"{cls.__name__}" object has no field "{name}"')
            elif config.get('extra') == 'allow' and name not in fields:
                if self.model_extra and name in self.model_extra:
                    self.__pydantic_extra__[name] = value  # type: ignore
                else:
                    try:
                        getattr(self, name)
                    except AttributeError:
                        self.__pydantic_extra__[name] = value  # type: ignore
                    else:
                        _object_setattr(self, name, value)
            else:
                self.__dict__[name] = value
                self.__pydantic_fields_set__.add(name)

'''

F1 = '''        def __setattr__(self, name: str, value: Any) -> None:
            handler = self.__pydantic_setattr_handlers__.get(name)
            if handler is not None:
                handler(self, name, value)
                return
            handler = self._v7_setattr_handler(name, value)
            if handler is not None:
                handler(self, name, value)
                self.__pydantic_setattr_handlers__[name] = handler

        def _v7_setattr_handler(self, name: str, value: Any) -> Callable[[BaseModel, str, Any], None] | None:
            cls = self.__class__
            if name in cls.__class_vars__:
                raise AttributeError(
                    f'{name!r} is a ClassVar of `{cls.__name__}` and cannot be set on an instance. '
                    f'If you want to set a value on the class, use `{cls.__name__}.{name} = value`.'
                )
            if not _fields.is_valid_field_name(name):
                private_attributes = cls.__private_attributes__
                if self.__pydantic_private__ is None or name not in private_attributes:
                    _object_setattr(self, name, value)
                    return None
                attribute = private_attributes[name]
                if hasattr(attribute, '__set__'):
                    return lambda model, _name, val: attribute.__set__(model, val)  # type: ignore
                def private_handler(model: BaseModel, attr_name: str, val: Any) -> None:
                    model.__pydantic_private__[attr_name] = val  # type: ignore[index]
                return private_handler

            self._check_frozen(name, value)
            attr = getattr(cls, name, None)
            if isinstance(attr, property):
                return lambda model, _name, val: attr.__set__(model, val)
            if isinstance(attr, cached_property):
                def cached_handler(model: BaseModel, attr_name: str, val: Any) -> None:
                    model.__dict__[attr_name] = val
                return cached_handler
            if cls.model_config.get('validate_assignment', None):
                def validate_handler(model: BaseModel, attr_name: str, val: Any) -> None:
                    model.__pydantic_validator__.validate_assignment(model, attr_name, val)
                return validate_handler

            fields = cls.__pydantic_fields__
            extra = cls.model_config.get('extra')
            if extra != 'allow' and name not in fields:
                raise ValueError(f'"{cls.__name__}" object has no field "{name}"')
            if extra == 'allow' and name not in fields:
                if self.model_extra and name in self.model_extra:
                    self.__pydantic_extra__[name] = value  # type: ignore
                else:
                    try:
                        getattr(self, name)
                    except AttributeError:
                        self.__pydantic_extra__[name] = value  # type: ignore
                    else:
                        _object_setattr(self, name, value)
                return None

            def field_handler(model: BaseModel, attr_name: str, val: Any) -> None:
                model.__dict__[attr_name] = val
                model.__pydantic_fields_set__.add(attr_name)
            return field_handler

'''

F2 = '''        def __setattr__(self, name: str, value: Any) -> None:
            handler = self.__pydantic_setattr_handlers__.get(name)
            if handler is not None:
                handler(self, name, value)
                return
            cls = self.__class__
            if name in cls.__class_vars__:
                raise AttributeError(
                    f'{name!r} is a ClassVar of `{cls.__name__}` and cannot be set on an instance. '
                    f'If you want to set a value on the class, use `{cls.__name__}.{name} = value`.'
                )
            elif not _fields.is_valid_field_name(name):
                private_attributes = cls.__private_attributes__
                if self.__pydantic_private__ is None or name not in private_attributes:
                    _object_setattr(self, name, value)
                    return
                attribute = private_attributes[name]
                if hasattr(attribute, '__set__'):
                    attribute.__set__(self, value)  # type: ignore
                    return
                def private_handler(model: BaseModel, attr_name: str, val: Any) -> None:
                    model.__pydantic_private__[attr_name] = val  # type: ignore[index]
                private_handler(self, name, value)
                self.__pydantic_setattr_handlers__[name] = private_handler
                return

            self._check_frozen(name, value)
            attr = getattr(cls, name, None)
            if isinstance(attr, property):
                attr.__set__(self, value)
                return
            if isinstance(attr, cached_property):
                def cached_handler(model: BaseModel, attr_name: str, val: Any) -> None:
                    model.__dict__[attr_name] = val
                cached_handler(self, name, value)
                self.__pydantic_setattr_handlers__[name] = cached_handler
                return
            if cls.model_config.get('validate_assignment', None):
                self.__pydantic_validator__.validate_assignment(self, name, value)
                return
            fields = cls.__pydantic_fields__
            extra = cls.model_config.get('extra')
            if extra != 'allow' and name not in fields:
                raise ValueError(f'"{cls.__name__}" object has no field "{name}"')
            if extra == 'allow' and name not in fields:
                if self.model_extra and name in self.model_extra:
                    self.__pydantic_extra__[name] = value  # type: ignore
                else:
                    try:
                        getattr(self, name)
                    except AttributeError:
                        self.__pydantic_extra__[name] = value  # type: ignore
                    else:
                        _object_setattr(self, name, value)
                return
            def field_handler(model: BaseModel, attr_name: str, val: Any) -> None:
                model.__dict__[attr_name] = val
                model.__pydantic_fields_set__.add(attr_name)
            field_handler(self, name, value)
            self.__pydantic_setattr_handlers__[name] = field_handler

'''


def ordinary_fast(require_extra_not_allow: bool) -> str:
    pred = "not config.get('validate_assignment', None)"
    if require_extra_not_allow:
        pred += " and config.get('extra') != 'allow'"
    return COMMON_HEAD + f'''            config = cls.model_config
            fields = cls.__pydantic_fields__
            if {pred} and name in fields:
                self.__dict__[name] = value
                self.__pydantic_fields_set__.add(name)
                return

''' + TAIL


def localized() -> str:
    return '''        def __setattr__(self, name: str, value: Any) -> None:
            cls = self.__class__
            class_vars = cls.__class_vars__
            private_attributes = cls.__private_attributes__
            config = cls.model_config
            fields = cls.__pydantic_fields__
            if name in class_vars:
                raise AttributeError(
                    f'{name!r} is a ClassVar of `{cls.__name__}` and cannot be set on an instance. '
                    f'If you want to set a value on the class, use `{cls.__name__}.{name} = value`.'
                )
            elif not _fields.is_valid_field_name(name):
                if self.__pydantic_private__ is None or name not in private_attributes:
                    _object_setattr(self, name, value)
                else:
                    attribute = private_attributes[name]
                    if hasattr(attribute, '__set__'):
                        attribute.__set__(self, value)  # type: ignore
                    else:
                        self.__pydantic_private__[name] = value
                return
            self._check_frozen(name, value)
''' + TAIL


def frozen_sets() -> str:
    return '''        def __setattr__(self, name: str, value: Any) -> None:
            cls = self.__class__
            field_names = frozenset(cls.__pydantic_fields__)
            private_names = frozenset(cls.__private_attributes__)
            if name in cls.__class_vars__:
                raise AttributeError(
                    f'{name!r} is a ClassVar of `{cls.__name__}` and cannot be set on an instance. '
                    f'If you want to set a value on the class, use `{cls.__name__}.{name} = value`.'
                )
            elif not _fields.is_valid_field_name(name):
                if self.__pydantic_private__ is None or name not in private_names:
                    _object_setattr(self, name, value)
                else:
                    attribute = cls.__private_attributes__[name]
                    if hasattr(attribute, '__set__'):
                        attribute.__set__(self, value)  # type: ignore
                    else:
                        self.__pydantic_private__[name] = value
                return
            self._check_frozen(name, value)
            config = cls.model_config
            fields = field_names
''' + TAIL


def original_local_tail(original: str) -> str:
    return original.replace(
        "            else:\n                self.__dict__[name] = value\n                self.__pydantic_fields_set__.add(name)\n",
        "            else:\n                model_dict = self.__dict__\n                fields_set = self.__pydantic_fields_set__\n                model_dict[name] = value\n                fields_set.add(name)\n",
    )


def apply(root: Path, candidate: str) -> None:
    main = root / 'pydantic/main.py'
    text = main.read_text(encoding='utf-8')
    start = text.index(START)
    end = text.index(END, start)
    original = text[start:end]
    bodies = {
        'F1': F1,
        'F2': F2,
        'F3': ordinary_fast(False),
        'N1': ordinary_fast(False),
        'N2': ordinary_fast(True),
        'N3': localized(),
        'R1': ordinary_fast(True),
        'R2': frozen_sets(),
        'R3': original_local_tail(original),
    }
    if candidate not in bodies:
        raise SystemExit(f'unknown candidate {candidate}')
    text = text[:start] + bodies[candidate] + text[end:]
    if candidate in {'F1', 'F2'}:
        text = text.replace(CLASSVAR_ANCHOR, CLASSVAR_INSERT + CLASSVAR_ANCHOR)
        model_construction = root / 'pydantic/_internal/_model_construction.py'
        mt = model_construction.read_text(encoding='utf-8')
        mt = mt.replace(META_ANCHOR, META_ANCHOR + "            cls.__pydantic_setattr_handlers__ = {}\n")
        model_construction.write_text(mt, encoding='utf-8')
    main.write_text(text, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    apply(args.root, args.candidate)


if __name__ == '__main__':
    main()
