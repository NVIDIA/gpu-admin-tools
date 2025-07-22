#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import importlib
import importlib.util
import os

class DeviceMetadata:
    def __init__(self, name: str, start_address: int, end_address: int):
        self.name = name
        self.start_address = start_address
        self.end_address = end_address

    def __str__(self):
        return f"{self.name} [0x{self.start_address:08x}-0x{self.end_address:08x}]"

class RegisterMetadata:
    """Metadata for a register (name, address, etc.)"""
    def __init__(self, name, address, priv_level_mask=None):
        self.name = name
        self.address = address
        self.fields = {}
        self.priv_level_mask = priv_level_mask  # Link to PRIV_LEVEL_MASK register if it exists

    def add_field(self, field):
        self.fields[field.name] = field

    def __str__(self):
        return f"{self.name} @ 0x{self.address:08x} ({len(self.fields)} fields)"

class FieldMetadata:
    """Metadata for a field (position, mask, etc.)"""
    def __init__(self, name, msb, lsb, register):
        self.name = name
        self.msb = msb
        self.lsb = lsb
        self.register = register
        self.mask = ((1 << (msb - lsb + 1)) - 1) << lsb
        self.values = {}

        # Add field to register
        if register:
            register.add_field(self)

    def add_value(self, field):
        self.values[field.name] = field

    def __str__(self):
        reg_name = self.register.name if self.register else "No register"
        return f"{self.name} [{self.msb}:{self.lsb}] in {reg_name} ({len(self.values)} values)"

class ValueMetadata:
    """Metadata for a specific field value"""
    def __init__(self, name, value, field):
        self.name = name
        self.value = value
        self.field = field

        # Add value to field
        if field:
            field.add_value(self)

    def __int__(self):
        """Allow conversion to int"""
        return self.value

    def __eq__(self, other):
        """Allow comparison with integers, FieldValue, and other ValueMetadata"""
        if isinstance(other, ValueMetadata):
            return self.value == other.value
        elif isinstance(other, FieldValue):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return NotImplemented

    def __str__(self):
        field_name = self.field.name if self.field else "No field"
        return f"{self.name} = 0x{self.value:x} in {field_name}"

class ArrayMetadata(RegisterMetadata):
    """Metadata for a register array that inherits from RegisterMetadata"""
    def __init__(self, name, base_address, stride, size, priv_level_mask=None):
        # Initialize the RegisterMetadata part with base_address
        super().__init__(name, base_address, priv_level_mask)

        # Add array-specific attributes
        self.stride = stride
        self.size = size

    def get_address(self, index):
        """Calculate address for a specific index"""
        if index < 0 or (self.size > 0 and index >= self.size):
            raise IndexError(f"Array index {index} out of bounds (0-{self.size-1})")
        return self.address + (index * self.stride)

    def __call__(self, index):
        """Get a register metadata for this array at index"""
        # Create a new register that points to the same fields
        reg = RegisterMetadata(
            name=f"{self.name}({index})",
            address=self.get_address(index),
            priv_level_mask=self.priv_level_mask
        )

        # Use the fields from this array directly
        reg.fields = self.fields

        return reg

    def __str__(self):
        size_info = f"{self.size} entries" if self.size > 0 else "unlimited"
        return f"{self.name} @ 0x{self.address:08x} (stride=0x{self.stride:x}, {size_info}, {len(self.fields)} fields)"

class RegisterValue:
    """The actual value of a register with methods to access fields"""
    def __init__(self, metadata, value=0):
        # Use object.__setattr__ to bypass our custom __setattr__
        object.__setattr__(self, 'metadata', metadata)
        object.__setattr__(self, 'value', value)
        object.__setattr__(self, '_allowed_attrs', {'metadata', 'value', '_allowed_attrs'})
        self._setup_field_properties()

    def _setup_field_properties(self):
        """Dynamically create properties for each field with register prefix removed"""
        # For array registers like "NV_INTERNAL", get base name before the parenthesis
        reg_name = self.metadata.name
        if '(' in reg_name:
            reg_prefix = reg_name[:reg_name.index('(')] + '_'
        else:
            reg_prefix = reg_name + '_'

        for field_name, field in self.metadata.fields.items():
            # Remove register prefix if present
            assert field_name.startswith(reg_prefix)

            short_name = field_name[len(reg_prefix):]

            assert short_name

            # Add field name to allowed attributes list
            self._allowed_attrs.add(short_name)

            # Create a property for this field with both getter and setter
            setattr(self.__class__, short_name, property(
                fget=lambda self, f=field: self.get_field_with_metadata(f),
                fset=lambda self, value, f=field: self._set_field(f, value),
                doc=f"Get or set the {short_name} field value"
            ))

    def _set_field(self, field_metadata, value):
        """Set a field's value in the register

        Args:
            field_metadata: The field to set
            value: Can be an integer, FieldValue, or ValueMetadata
        """
        # Handle ValueMetadata directly
        if isinstance(value, ValueMetadata):
            # Verify the field matches if both have a field reference
            if field_metadata and value.field and field_metadata != value.field:
                raise ValueError(f"Field mismatch: trying to set {field_metadata.name} with value for {value.field.name}")
            int_value = value.value
        else:
            # Convert other types to integer
            int_value = int(value)

        # Clear the field bits and set the new value
        self.value = (self.value & ~field_metadata.mask) | ((int_value << field_metadata.lsb) & field_metadata.mask)

        return self

    def get_field(self, field_metadata):
        """Extract a specific field from this register value"""
        return (self.value & field_metadata.mask) >> field_metadata.lsb

    def get_field_with_metadata(self, field_metadata):
        """Get field value with additional metadata (named values)"""
        field_value = self.get_field(field_metadata)

        # Check if this value has a name
        for val_name, val in field_metadata.values.items():
            if val.value == field_value:
                return FieldValue(field_metadata, field_value, val_name)

        # Always return a FieldValue for consistency
        return FieldValue(field_metadata, field_value)

    def get_field_by_name(self, field_name):
        """Get a field value by its name"""
        if field_name in self.metadata.fields:
            return self.get_field_with_metadata(self.metadata.fields[field_name])

        # Try with register prefix removed
        reg_prefix = self.metadata.name.split('_')[0] + '_'
        full_field_name = f"{reg_prefix}{field_name}"
        if full_field_name in self.metadata.fields:
            return self.get_field_with_metadata(self.metadata.fields[full_field_name])

        raise ValueError(f"Field '{field_name}' not found in register {self.metadata.name}")

    def __setattr__(self, name, value):
        """Control attribute assignment to prevent typos or unauthorized attributes"""
        if name in self._allowed_attrs or hasattr(self.__class__, name):
            # If it's an allowed attribute or a property/method of the class, set it
            object.__setattr__(self, name, value)
        else:
            # Try to match with an existing field as a fallback
            try:
                # See if this might be a field or a prefix of a field
                self.get_field_by_name(name)
                # If we found it, set it using the property
                # This will call the property setter which calls _set_field
                super().__setattr__(name, value)
            except ValueError:
                # Not an allowed attribute or field
                raise AttributeError(f"Cannot set '{name}' on RegisterValue. Did you mean one of: "
                                    f"{', '.join(sorted(self._allowed_attrs))}")

    def __str__(self):
        """Pretty print the register value with fields"""
        result = [f"{self.metadata.name} {self.metadata.address:#x} = 0x{self.value:08X} ({self.value})"]
        for field_name, field in self.metadata.fields.items():
            field_value = self.get_field(field)
            # Find named value if exists
            value_name = None
            for val_name, val in field.values.items():
                if val.value == field_value:
                    value_name = val_name
                    break

            if value_name:
                result.append(f"  {field_name} = {value_name} (0x{field_value:X})")
            else:
                result.append(f"  {field_name} = 0x{field_value:X}")

        return "\n".join(result)

    def __int__(self):
        """Convert to integer - allows `int(reg_value)`"""
        return self.value

    def __eq__(self, other):
        """Allow direct comparison with integers"""
        if isinstance(other, (int, RegisterValue)):
            return self.value == int(other)
        return NotImplemented

    def __and__(self, other):
        """Bitwise AND: reg_value & other"""
        return self.value & int(other)

    def __or__(self, other):
        """Bitwise OR: reg_value | other"""
        return self.value | int(other)

    def __xor__(self, other):
        """Bitwise XOR: reg_value ^ other"""
        return self.value ^ int(other)

    def __lshift__(self, other):
        """Left shift: reg_value << other"""
        return self.value << int(other)

    def __rshift__(self, other):
        """Right shift: reg_value >> other"""
        return self.value >> int(other)

    # Right-hand versions for when the RegisterValue is on the right
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

class FieldValue:
    """Represents a field value with both numeric value and metadata"""
    def __init__(self, field, value, name=None):
        self.field = field  # Reference to the field metadata (required)
        self.value = value  # The numeric value
        self.name = name    # Optional name for the value

    def __int__(self):
        return self.value

    def __eq__(self, other):
        """Enhanced equality comparison with field validation"""
        if isinstance(other, FieldValue):
            # Different fields means values aren't comparable
            if self.field != other.field:
                return False
            return self.value == other.value
        elif isinstance(other, ValueMetadata):
            # Different fields means values aren't comparable
            if other.field and self.field != other.field:
                return False
            return self.value == other.value
        return self.value == other

    def __str__(self):
        """String representation with field information"""
        if self.name:
            return f"{self.name} ({self.field.name}: 0x{self.value:X})"
        return f"{self.field.name}: 0x{self.value:X}"


class LazyModuleDescriptor:
    """Descriptor for lazy-loaded modules"""
    def __init__(self, module_path):
        self.module_path = module_path
        self.module = None

    def __get__(self, instance, owner):
        if self.module is None:
            self.module = importlib.import_module(self.module_path)
        return self.module

class RegisterInterface:
    """Interface for accessing registers on a GPU"""
    def __init__(self, gpu):
        self.gpu = gpu
        self._setup_lazy_modules()

    def _setup_lazy_modules(self):
        """Discover and set up lazy-loaded module attributes"""
        chip = self.gpu.chip
        if chip.startswith("gb2"):
            # Use gb202 for gb20x
            chip = "gb202"
        elif chip == "gb110":
            chip = "gb100"
        elif chip == "gb112":
            chip = "gb102"

        modules = self._discover_modules(chip)

        # Set up lazy loading for each available module
        for module_name in modules:
            # Create the full module path
            module_path = f"gpu.regs.{chip}.{module_name}"

            # Check if the module exists
            if self._module_exists(module_path):
                # Set up a lazy-loaded descriptor for this module
                setattr(self.__class__, module_name, LazyModuleDescriptor(module_path))

    def _discover_modules(self, chip):
        """Discover available modules for the given chip"""
        modules = set()

        # Import the chip's package
        chip_package = importlib.import_module(f"gpu.regs.{chip}")
        package_path = os.path.dirname(chip_package.__file__)

        # Scan for Python files in the package directory
        for item in os.listdir(package_path):
            if item.endswith('.py') and not item.startswith('__'):
                module_name = item[:-3]  # Remove .py extension
                modules.add(module_name)

        return modules

    def _module_exists(self, module_path):
        """Check if a module exists without importing it"""
        try:
            spec = importlib.util.find_spec(module_path)
            return spec is not None
        except (ImportError, AttributeError):
            return False

    def read(self, register_or_field):
        """Read a register or field from GPU"""
        if isinstance(register_or_field, RegisterMetadata):
            reg_value = self.gpu.read_bad_ok(register_or_field.address)
            return RegisterValue(register_or_field, reg_value)
        elif isinstance(register_or_field, FieldMetadata):
            reg = self.read(register_or_field.register)
            return reg.get_field(register_or_field)
        else:
            raise TypeError(f"Can't read from {type(register_or_field)}")

    def write(self, register, value):
        """Write a value to a register"""
        if isinstance(register, RegisterMetadata):
            self.gpu.write(register.address, value)
        else:
            raise TypeError(f"Can't write to {type(register)}")

    def write_field(self, field, value):
        """Write a value to a specific field, preserving other fields"""
        if isinstance(field, FieldMetadata):
            reg_value = self.gpu.read(field.register.address)
            # Clear field bits and set new value
            new_value = (reg_value & ~field.mask) | ((value << field.lsb) & field.mask)
            self.gpu.write(field.register.address, new_value)
        else:
            raise TypeError(f"Can't write to field {type(field)}")

    def is_set(self, value_metadata):
        """Check if a field has the specified value"""
        if isinstance(value_metadata, ValueMetadata):
            field_value = self.read(value_metadata.field)
            return field_value == value_metadata.value
        else:
            raise TypeError(f"Can't check value {type(value_metadata)}")
