# backend/code_validator.py
"""
Code Validator - Validates generated React code
Checks for syntax errors, component usage, and best practices
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class CodeValidator:
    """Validates generated React code"""
    
    # Allowed component library
    ALLOWED_COMPONENTS = [
        'Button', 'Card', 'Input', 'Table', 'Chart',
        'Navbar', 'Sidebar', 'Modal'
    ]
    
    # Required imports for each component
    REQUIRED_IMPORTS = {
        'Button': "import Button from '@/components/ui/Button';",
        'Card': "import Card from '@/components/ui/Card';",
        'Input': "import Input from '@/components/ui/Input';",
        'Table': "import Table from '@/components/ui/Table';",
        'Chart': "import Chart from '@/components/ui/Chart';",
        'Navbar': "import Navbar from '@/components/ui/Navbar';",
        'Sidebar': "import Sidebar from '@/components/ui/Sidebar';",
        'Modal': "import Modal from '@/components/ui/Modal';",
    }
    
    def __init__(self):
        """Initialize the validator"""
        pass
    
    def validate(self, code: str) -> ValidationResult:
        """
        Validate generated React code
        
        Args:
            code: Generated React code string
            
        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check 1: Basic syntax
        syntax_errors = self._check_syntax(code)
        errors.extend(syntax_errors)
        
        # Check 2: Component usage
        component_errors = self._check_components(code)
        errors.extend(component_errors)
        
        # Check 3: Imports
        import_warnings = self._check_imports(code)
        warnings.extend(import_warnings)
        
        # Check 4: Props validation
        prop_warnings = self._check_props(code)
        warnings.extend(prop_warnings)
        
        # Check 5: Best practices
        best_practice_suggestions = self._check_best_practices(code)
        suggestions.extend(best_practice_suggestions)
        
        # Check 6: Accessibility
        a11y_suggestions = self._check_accessibility(code)
        suggestions.extend(a11y_suggestions)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _check_syntax(self, code: str) -> List[str]:
        """Check for basic syntax errors"""
        errors = []
        
        # Check for unclosed JSX tags
        open_tags = re.findall(r'<(\w+)[^/>]*>', code)
        close_tags = re.findall(r'</(\w+)>', code)
        self_closing = re.findall(r'<(\w+)[^>]*/>', code)
        
        # Remove self-closing tags from open tags
        for tag in self_closing:
            if tag in open_tags:
                open_tags.remove(tag)
        
        # Check if all tags are closed
        if len(open_tags) != len(close_tags):
            errors.append(f"Mismatched JSX tags: {len(open_tags)} open, {len(close_tags)} close")
        
        # Check for unclosed braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            errors.append(f"Mismatched braces: {open_braces} open, {close_braces} close")
        
        # Check for unclosed parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            errors.append(f"Mismatched parentheses: {open_parens} open, {close_parens} close")
        
        # Check for export default
        if 'export default' not in code:
            errors.append("Missing 'export default' statement")
        
        return errors
    
    def _check_components(self, code: str) -> List[str]:
        """Check if only allowed components are used"""
        errors = []
        
        # Find all component usages
        components_used = re.findall(r'<(\w+)', code)
        
        # Filter out HTML elements
        html_elements = ['div', 'span', 'p', 'h1', 'h2', 'h3', 'section', 'article']
        components_used = [c for c in components_used if c not in html_elements]
        
        # Check if all components are allowed
        for component in set(components_used):
            # Remove sub-components (e.g., Card.Title -> Card)
            base_component = component.split('.')[0]
            
            if base_component not in self.ALLOWED_COMPONENTS:
                errors.append(f"Unauthorized component used: {component}")
        
        return errors
    
    def _check_imports(self, code: str) -> List[str]:
        """Check if required imports are present"""
        warnings = []
        
        # Find components used
        components_used = re.findall(r'<(\w+)', code)
        components_used = [c.split('.')[0] for c in components_used]
        
        # Check if imports are present for each component
        for component in set(components_used):
            if component in self.ALLOWED_COMPONENTS:
                required_import = self.REQUIRED_IMPORTS[component]
                if required_import not in code:
                    warnings.append(f"Missing import for component: {component}")
        
        return warnings
    
    def _check_props(self, code: str) -> List[str]:
        """Check for common prop errors"""
        warnings = []
        
        # Check for inline styles (not allowed)
        if 'style={{' in code or 'style={' in code:
            warnings.append("Inline styles detected - use Tailwind classes instead")
        
        # Check for className presence
        if 'className=' not in code:
            warnings.append("No className detected - consider adding Tailwind classes")
        
        return warnings
    
    def _check_best_practices(self, code: str) -> List[str]:
        """Check for React best practices"""
        suggestions = []
        
        # Check for key prop in lists (if any map is present)
        if '.map(' in code and 'key=' not in code:
            suggestions.append("Consider adding 'key' prop when rendering lists")
        
        # Check for component naming
        if not re.search(r'export default function [A-Z]\w+', code):
            suggestions.append("Component name should be PascalCase")
        
        # Check for prop destructuring
        if 'function' in code and 'props.' in code:
            suggestions.append("Consider destructuring props in function signature")
        
        return suggestions
    
    def _check_accessibility(self, code: str) -> List[str]:
        """Check for accessibility issues"""
        suggestions = []
        
        # Check for buttons without accessible text
        buttons = re.findall(r'<Button[^>]*>', code)
        for button in buttons:
            if 'aria-label' not in button and '>' not in button:
                suggestions.append("Add aria-label or text content to buttons")
        
        # Check for images without alt text
        if '<img' in code and 'alt=' not in code:
            suggestions.append("Add alt text to images for accessibility")
        
        return suggestions
    
    def fix_common_issues(self, code: str) -> str:
        """Automatically fix common issues in code"""
        fixed_code = code
        
        # Fix missing semicolons in imports
        fixed_code = re.sub(r'(import .+)(?<!;)$', r'\1;', fixed_code, flags=re.MULTILINE)
        
        # Ensure proper spacing
        fixed_code = re.sub(r'([>}])\s*([<{])', r'\1\n\2', fixed_code)
        
        return fixed_code


# Example usage
if __name__ == '__main__':
    validator = CodeValidator()
    
    # Test code
    test_code = """import Button from '@/components/ui/Button';
import Card from '@/components/ui/Card';

export default function GeneratedComponent() {
  return (
    <div className="grid grid-cols-2 gap-4">
      <Card>
        <Card.Title>Card 1</Card.Title>
      </Card>
      <Button variant="primary">Click me</Button>
    </div>
  );
}"""
    
    result = validator.validate(test_code)
    
    print("Validation Result:")
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Suggestions: {result.suggestions}")