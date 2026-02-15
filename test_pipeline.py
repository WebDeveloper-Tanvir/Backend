# backend/test_pipeline.py
"""
Test the complete pipeline end-to-end
Run this to see how it works without starting the server
"""

# --- BEGIN INLINED intent_parser.py content ---
import re
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Intent:
    """Represents the parsed user intent"""
    primary_action: str  # create, add, modify, remove
    ui_type: str  # dashboard, form, table, card, etc.
    components: List[str]  # List of component names
    layout: str  # grid, flex, stack, etc.
    modifiers: Dict[str, any]  # Additional properties
    confidence: float  # 0-1 confidence score

class IntentParser:
    """Parses user input to determine what UI to build"""

    ACTIONS = {
        'create': ['create', 'make', 'build', 'generate', 'new'],
        'add': ['add', 'include', 'insert', 'put'],
        'modify': ['change', 'update', 'modify', 'edit', 'alter'],
        'remove': ['remove', 'delete', 'take out', 'get rid of'],
    }

    UI_TYPES = {
        'dashboard': ['dashboard', 'overview', 'summary', 'analytics'],
        'form': ['form', 'input form', 'registration', 'signup', 'login'],
        'table': ['table', 'data table', 'grid', 'list'],
        'card': ['card', 'info card', 'profile card'],
        'navbar': ['navbar', 'navigation', 'menu', 'header'],
        'sidebar': ['sidebar', 'side menu', 'drawer'],
        'modal': ['modal', 'dialog', 'popup', 'overlay'],
        'button': ['button', 'btn', 'action button'],
        'chart': ['chart', 'graph', 'visualization', 'plot'],
    }

    COMPONENTS = {
        'Button': ['button', 'btn', 'submit', 'action'],
        'Card': ['card', 'panel', 'box'],
        'Input': ['input', 'textbox', 'field', 'text field'],
        'Table': ['table', 'data table', 'grid'],
        'Chart': ['chart', 'graph', 'bar chart', 'line chart', 'pie chart'],
        'Navbar': ['navbar', 'navigation', 'top bar', 'header'],
        'Sidebar': ['sidebar', 'side panel', 'drawer'],
        'Modal': ['modal', 'dialog', 'popup'],
    }

    LAYOUTS = {
        'grid': ['grid', 'tiles', 'cards'],
        'flex': ['flex', 'flexible', 'row', 'column'],
        'stack': ['stack', 'vertical', 'horizontal'],
        'split': ['split', 'divided', 'sections'],
    }

    VARIANTS = {
        'primary': ['primary', 'main', 'important'],
        'secondary': ['secondary', 'alternate'],
        'outline': ['outline', 'outlined', 'border'],
        'ghost': ['ghost', 'transparent', 'minimal'],
    }

    COLORS = {
        'blue': ['blue'],
        'red': ['red', 'danger', 'error'],
        'green': ['green', 'success'],
        'yellow': ['yellow', 'warning'],
        'gray': ['gray', 'grey', 'neutral'],
    }

    def __init__(self):
        pass

    def parse(self, user_input: str) -> Intent:
        user_input = user_input.lower().strip()
        action = self._extract_action(user_input)
        ui_type = self._extract_ui_type(user_input)
        components = self._extract_components(user_input)
        layout = self._extract_layout(user_input)
        modifiers = self._extract_modifiers(user_input)
        confidence = self._calculate_confidence(action, ui_type, components)
        return Intent(
            primary_action=action,
            ui_type=ui_type,
            components=components,
            layout=layout,
            modifiers=modifiers,
            confidence=confidence
        )

    def _extract_action(self, text: str) -> str:
        for action, keywords in self.ACTIONS.items():
            if any(keyword in text for keyword in keywords):
                return action
        return 'create'

    def _extract_ui_type(self, text: str) -> str:
        for ui_type, keywords in self.UI_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return ui_type
        if 'input' in text and 'button' in text:
            return 'form'
        elif 'card' in text or 'kpi' in text:
            return 'dashboard'
        return 'generic'

    def _extract_components(self, text: str) -> List[str]:
        found_components = []
        for component, keywords in self.COMPONENTS.items():
            if any(keyword in text for keyword in keywords):
                if component not in found_components:
                    found_components.append(component)
        if not found_components:
            found_components = self._infer_components_from_ui_type(
                self._extract_ui_type(text)
            )
        return found_components

    def _extract_layout(self, text: str) -> str:
        for layout, keywords in self.LAYOUTS.items():
            if any(keyword in text for keyword in keywords):
                return layout
        return 'flex'

    def _extract_modifiers(self, text: str) -> Dict[str, any]:
        modifiers = {}
        for variant, keywords in self.VARIANTS.items():
            if any(keyword in text for keyword in keywords):
                modifiers['variant'] = variant
                break
        for color, keywords in self.COLORS.items():
            if any(keyword in text for keyword in keywords):
                modifiers['color'] = color
                break
        if 'large' in text or 'big' in text:
            modifiers['size'] = 'large'
        elif 'small' in text or 'tiny' in text:
            modifiers['size'] = 'small'
        numbers = re.findall(r'\\b(\\d+)\\b', text)
        if numbers:
            modifiers['count'] = int(numbers[0])
        return modifiers

    def _infer_components_from_ui_type(self, ui_type: str) -> List[str]:
        defaults = {
            'dashboard': ['Card', 'Chart'],
            'form': ['Input', 'Button'],
            'table': ['Table'],
            'card': ['Card'],
            'navbar': ['Navbar'],
            'sidebar': ['Sidebar'],
            'modal': ['Modal', 'Button'],
            'button': ['Button'],
            'chart': ['Chart'],
        }
        return defaults.get(ui_type, ['Card'])

    def _calculate_confidence(self, action: str, ui_type: str,
                            components: List[str]) -> float:
        score = 0.5
        if action != 'create':
            score += 0.1
        if ui_type != 'generic':
            score += 0.2
        if components:
            score += 0.2
        return min(score, 1.0)
# --- END INLINED intent_parser.py content ---

# --- BEGIN INLINED planner.py content ---
from typing import Any # Already imported Dict, List, dataclass

@dataclass
class ComponentPlan:
    """Plan for a single component"""
    type: str
    props: Dict[str, Any]
    children: List['ComponentPlan']
    position: Dict[str, int]  # row, col for grid layout

@dataclass
class UIPlan:
    """Complete UI generation plan"""
    layout_type: str
    container_props: Dict[str, Any]
    components: List[ComponentPlan]
    imports: List[str]
    reasoning: str

class Planner:
    """Creates structured plans from user intent"""

    TEMPLATES = {
        'dashboard': {
            'layout': 'grid',
            'container': {'className': 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-6'},
            'default_components': [
                {'type': 'Card', 'count': 3, 'variant': 'stats'},
                {'type': 'Chart', 'count': 2, 'variant': 'line'},
            ]
        },
        'form': {
            'layout': 'stack',
            'container': {'className': 'max-w-md mx-auto p-6 space-y-4'},
            'default_components': [
                {'type': 'Input', 'count': 3},
                {'type': 'Button', 'count': 1, 'variant': 'primary'},
            ]
        },
        'table': {
            'layout': 'flex',
            'container': {'className': 'w-full p-6'},
            'default_components': [
                {'type': 'Table', 'count': 1},
            ]
        },
        'navbar': {
            'layout': 'flex',
            'container': {'className': 'w-full'},
            'default_components': [
                {'type': 'Navbar', 'count': 1},
            ]
        },
    }

    def __init__(self):
        pass

    def create_plan(self, intent: Intent) -> UIPlan:
        template = self.TEMPLATES.get(intent.ui_type, self.TEMPLATES['form'])
        container_props = template['container'].copy()
        components = self._plan_components(intent, template)
        imports = self._determine_imports(components)
        reasoning = self._generate_reasoning(intent, components)
        return UIPlan(
            layout_type=template['layout'],
            container_props=container_props,
            components=components,
            imports=imports,
            reasoning=reasoning
        )

    def _plan_components(self, intent: Intent, template: Dict) -> List[ComponentPlan]:
        planned_components = []
        if intent.components:
            for idx, component_type in enumerate(intent.components):
                component = self._create_component_plan(
                    component_type,
                    intent.modifiers,
                    position={'row': idx // 3, 'col': idx % 3}
                )
                planned_components.append(component)
        else:
            for comp_def in template.get('default_components', []):
                count = comp_def.get('count', 1)
                for i in range(count):
                    component = self._create_component_plan(
                        comp_def['type'],
                        {'variant': comp_def.get('variant', 'default')},
                        position={'row': i // 3, 'col': i % 3}
                    )
                    planned_components.append(component)
        return planned_components

    def _create_component_plan(self, component_type: str, modifiers: Dict,
                               position: Dict) -> ComponentPlan:
        base_props = {
            'Button': {
                'variant': modifiers.get('variant', 'primary'),
                'children': 'Click me',
            },
            'Card': {
                'title': 'Card Title',
                'className': 'p-4',
            },
            'Input': {
                'label': 'Input Label',
                'placeholder': 'Enter value...',
            },
            'Table': {
                'columns': ['Name', 'Email', 'Role', 'Status'],
                'data': [
                    {'name': 'John Doe', 'email': 'john@example.com', 'role': 'Admin', 'status': 'Active'},
                    {'name': 'Jane Smith', 'email': 'jane@example.com', 'role': 'User', 'status': 'Active'},
                ],
            },
            'Chart': {
                'type': 'line',
                'data': [
                    {'name': 'Jan', 'value': 400},
                    {'name': 'Feb', 'value': 300},
                    {'name': 'Mar', 'value': 600},
                ],
            },
            'Navbar': {
                'brand': 'My App',
                'links': ['Home', 'About', 'Contact'],
            },
            'Sidebar': {
                'items': ['Dashboard', 'Profile', 'Settings'],
            },
            'Modal': {
                'title': 'Modal Title',
                'children': 'Modal content goes here',
            },
        }
        props = base_props.get(component_type, {}).copy()
        if 'variant' in modifiers:
            props['variant'] = modifiers['variant']
        if 'color' in modifiers:
            props['color'] = modifiers['color']
        if 'size' in modifiers:
            props['size'] = modifiers['size']
        return ComponentPlan(
            type=component_type,
            props=props,
            children=[],
            position=position
        )

    def _determine_imports(self, components: List[ComponentPlan]) -> List[str]:
        component_types = set(comp.type for comp in components)
        imports = [
            f"import {comp_type} from '@/components/ui/{comp_type}';"
            for comp_type in component_types
        ]
        if 'Chart' in component_types:
            imports.append("import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';")
        return imports

    def _generate_reasoning(self, intent: Intent,
                           components: List[ComponentPlan]) -> str:
        component_names = [comp.type for comp in components]
        unique_components = list(set(component_names))
        reasoning = f"Created a {intent.ui_type} layout using "
        reasoning += f"{len(components)} component(s): {', '.join(unique_components)}. "
        if intent.layout:
            reasoning += f"Arranged in a {intent.layout} layout. "
        if intent.modifiers:
            reasoning += f"Applied modifiers: {intent.modifiers}."
        return reasoning
# --- END INLINED planner.py content ---

# --- BEGIN INLINED code_generator.py content ---
# from typing import Dict, List, Any # Already imported
# import json # Already imported

class CodeGenerator:
    """Generates React code from UI plans"""

    def __init__(self):
        pass

    def generate(self, plan: UIPlan) -> str:
        imports = self._generate_imports(plan.imports)
        component_code = self._generate_component(plan)
        full_code = f"{imports}\n\n{component_code}"
        return full_code

    def _generate_imports(self, imports: List[str]) -> str:
        if not imports:
            return ""
        return "\n".join(imports)

    def _generate_component(self, plan: UIPlan) -> str:
        components_jsx = []
        for comp_plan in plan.components:
            jsx = self._generate_component_jsx(comp_plan)
            components_jsx.append(jsx)
        container_class = plan.container_props.get('className', '')
        code = f"""export default function GeneratedComponent() {{
  return (
    <div className=\"{container_class}\">
{self._indent(chr(10).join(components_jsx), 6)}
    </div>
  );
}}"""
        return code

    def _generate_component_jsx(self, comp_plan: ComponentPlan) -> str:
        comp_type = comp_plan.type
        props = comp_plan.props
        if comp_type == 'Button':
            return self._generate_button(props)
        elif comp_type == 'Card':
            return self._generate_card(props)
        elif comp_type == 'Input':
            return self._generate_input(props)
        elif comp_type == 'Table':
            return self._generate_table(props)
        elif comp_type == 'Chart':
            return self._generate_chart(props)
        elif comp_type == 'Navbar':
            return self._generate_navbar(props)
        elif comp_type == 'Sidebar':
            return self._generate_sidebar(props)
        elif comp_type == 'Modal':
            return self._generate_modal(props)
        else:
            return f"<div>Unsupported component: {comp_type}</div>"

    def _generate_button(self, props: Dict) -> str:
        variant = props.get('variant', 'primary')
        children = props.get('children', 'Button')
        return f'<Button variant=\"{variant}\">{children}</Button>'

    def _generate_card(self, props: Dict) -> str:
        title = props.get('title', 'Card Title')
        content = props.get('content', 'Card content goes here.')
        return f'''<Card>
  <Card.Title>{title}</Card.Title>
  <Card.Content>
    <p>{content}</p>
  </Card.Content>
</Card>'''

    def _generate_input(self, props: Dict) -> str:
        label = props.get('label', 'Label')
        placeholder = props.get('placeholder', 'Enter value...')
        return f'<Input label=\"{label}\" placeholder=\"{placeholder}\" />'

    def _generate_table(self, props: Dict) -> str:
        columns = props.get('columns', ['Column 1', 'Column 2'])
        data = props.get('data', [])
        col_defs = ', '.join([f'{{ header: \"{col}\", accessor: \"{col.lower().replace(" ", "_")}\" }}'
                             for col in columns])
        data_str = json.dumps(data, indent=2)
        return f'''<Table>
  columns={{[{col_defs}]}}
  data={{data}}
/>'''

    def _generate_chart(self, props: Dict) -> str:
        chart_type = props.get('type', 'line')
        data = props.get('data', [])
        data_str = json.dumps(data, indent=2)
        if chart_type == 'line':
            return f'''<Chart type=\"line\" data={{{data_str}}} />'''
        elif chart_type == 'bar':
            return f'''<Chart type=\"bar\" data={{{data_str}}} />'''
        else:
            return f'''<Chart type=\"line\" data={{{data_str}}} />'''

    def _generate_navbar(self, props: Dict) -> str:
        brand = props.get('brand', 'Brand')
        links = props.get('links', ['Home', 'About'])
        return f'''<Navbar brand=\"{brand}\">
  {' '.join([f'<Navbar.Link>{link}</Navbar.Link>' for link in links])}
</Navbar>'''

    def _generate_sidebar(self, props: Dict) -> str:
        items = props.get('items', ['Item 1', 'Item 2'])
        items_jsx = '\n  '.join([f'<Sidebar.Item>{item}</Sidebar.Item>' for item in items])
        return f'''<Sidebar>
  {items_jsx}
</Sidebar>'''

    def _generate_modal(self, props: Dict) -> str:
        title = props.get('title', 'Modal Title')
        children = props.get('children', 'Modal content')
        return f'''<Modal>
  <Modal.Title>{title}</Modal.Title>
  <Modal.Content>
    <p>{children}</p>
  </Modal.Content>
  <Modal.Footer>
    <Button variant=\"primary\">Save</Button>
    <Button variant=\"secondary\">Cancel</Button>
  </Modal.Footer>
</Modal>'''

    def _indent(self, text: str, spaces: int) -> str:
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line
                        for line in text.split('\n'))
# --- END INLINED code_generator.py content ---

# --- BEGIN INLINED code_validator.py content ---
# from typing import Dict, List, Tuple # Already imported Dict, List
# from dataclasses import dataclass # Already imported dataclass

@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class CodeValidator:
    """Validates generated React code"""

    ALLOWED_COMPONENTS = [
        'Button', 'Card', 'Input', 'Table', 'Chart',
        'Navbar', 'Sidebar', 'Modal'
    ]

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
        pass

    def validate(self, code: str) -> ValidationResult:
        errors = []
        warnings = []
        suggestions = []
        syntax_errors = self._check_syntax(code)
        errors.extend(syntax_errors)
        component_errors = self._check_components(code)
        errors.extend(component_errors)
        import_warnings = self._check_imports(code)
        warnings.extend(import_warnings)
        prop_warnings = self._check_props(code)
        warnings.extend(prop_warnings)
        best_practice_suggestions = self._check_best_practices(code)
        suggestions.extend(best_practice_suggestions)
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
        errors = []
        open_tags = re.findall(r'<(\w+)[^/>]*>', code)
        close_tags = re.findall(r'</(\w+)>', code)
        self_closing = re.findall(r'<(\w+)[^>]*/>', code)
        for tag in self_closing:
            if tag in open_tags:
                open_tags.remove(tag)
        if len(open_tags) != len(close_tags):
            errors.append(f"Mismatched JSX tags: {len(open_tags)} open, {len(close_tags)} close")
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            errors.append(f"Mismatched braces: {open_braces} open, {close_braces} close")
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            errors.append(f"Mismatched parentheses: {open_parens} open, {close_parens} close")
        if 'export default' not in code:
            errors.append("Missing 'export default' statement")
        return errors

    def _check_components(self, code: str) -> List[str]:
        errors = []
        components_used = re.findall(r'<(\w+)', code)
        html_elements = ['div', 'span', 'p', 'h1', 'h2', 'h3', 'section', 'article']
        components_used = [c for c in components_used if c not in html_elements]
        for component in set(components_used):
            base_component = component.split('.')[0]
            if base_component not in self.ALLOWED_COMPONENTS:
                errors.append(f"Unauthorized component used: {component}")
        return errors

    def _check_imports(self, code: str) -> List[str]:
        warnings = []
        components_used = re.findall(r'<(\w+)', code)
        components_used = [c.split('.')[0] for c in components_used]
        for component in set(components_used):
            if component in self.ALLOWED_COMPONENTS:
                required_import = self.REQUIRED_IMPORTS[component]
                if required_import not in code:
                    warnings.append(f"Missing import for component: {component}")
        return warnings

    def _check_props(self, code: str) -> List[str]:
        warnings = []
        if 'style={{' in code or 'style={' in code:
            warnings.append("Inline styles detected - use Tailwind classes instead")
        if 'className=' not in code:
            warnings.append("No className detected - consider adding Tailwind classes")
        return warnings

    def _check_best_practices(self, code: str) -> List[str]:
        suggestions = []
        if '.map(' in code and 'key=' not in code:
            suggestions.append("Consider adding 'key' prop when rendering lists")
        if not re.search(r'export default function [A-Z]\w+', code):
            suggestions.append("Component name should be PascalCase")
        if 'function' in code and 'props.' in code:
            suggestions.append("Consider destructuring props in function signature")
        return suggestions

    def _check_accessibility(self, code: str) -> List[str]:
        suggestions = []
        buttons = re.findall(r'<Button[^>]*>', code)
        for button in buttons:
            if 'aria-label' not in button and '>' not in button:
                suggestions.append("Add aria-label or text content to buttons")
        if '<img' in code and 'alt=' not in code:
            suggestions.append("Add alt text to images for accessibility")
        return suggestions

    def fix_common_issues(self, code: str) -> str:
        fixed_code = code
        fixed_code = re.sub(r'(import .+)(?<!;)$', r'\\1;', fixed_code, flags=re.MULTILINE)
        fixed_code = re.sub(r'([>}])\\s*([<{])', r'\\1\\n\\2', fixed_code)
        return fixed_code
# --- END INLINED code_validator.py content ---

def test_pipeline(prompt: str):
    """Test the complete pipeline with a prompt"""
    print("=" * 70)
    print(f"📝 PROMPT: {prompt}")
    print("=" * 70)

    # Initialize components
    parser = IntentParser()
    planner = Planner()
    generator = CodeGenerator()
    validator = CodeValidator()

    # Step 1: Parse Intent
    print("\n1️⃣  PARSING INTENT...")
    intent = parser.parse(prompt)
    print(f"   Action: {intent.primary_action}")
    print(f"   UI Type: {intent.ui_type}")
    print(f"   Components: {intent.components}")
    print(f"   Layout: {intent.layout}")
    print(f"   Modifiers: {intent.modifiers}")
    print(f"   Confidence: {intent.confidence:.2f}")

    # Step 2: Create Plan
    print("\n2️⃣  CREATING PLAN...")
    plan = planner.create_plan(intent)
    print(f"   Layout Type: {plan.layout_type}")
    print(f"   Components Planned: {len(plan.components)}")
    print(f"   Reasoning: {plan.reasoning}")

    # Step 3: Generate Code
    print("\n3️⃣  GENERATING CODE...")
    code = generator.generate(plan)
    print(f"   Code Length: {len(code)} characters")
    print(f"   Lines: {len(code.split(chr(10)))} lines")

    # Step 4: Validate Code
    print("\n4️⃣  VALIDATING CODE...")
    validation = validator.validate(code)
    print(f"   Valid: {validation.is_valid}")
    print(f"   Errors: {len(validation.errors)}")
    print(f"   Warnings: {len(validation.warnings)}")
    print(f"   Suggestions: {len(validation.suggestions)}")

    if validation.errors:
        print("\n   ⚠️  Errors:")
        for error in validation.errors:
            print(f"      - {error}")

    if validation.warnings:
        print("\n   ⚠️  Warnings:")
        for warning in validation.warnings:
            print(f"      - {warning}")

    # Display Generated Code
    print("\n" + "=" * 70)
    print("✅ GENERATED CODE:")
    print("=" * 70)
    print(code)
    print("=" * 70)

    return code, validation


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Create a dashboard with 3 cards and 2 charts",
        "Build a login form with email and password",
        "Make a data table with 5 columns",
        "Create a navbar with logo and menu",
        "Add a modal with title and buttons",
    ]

    print("\n" + "🚀" * 35)
    print("RULE-BASED UI GENERATOR - PIPELINE TEST")
    print("🚀" * 35 + "\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'*' * 70}")
        print(f"TEST CASE {i}/{len(test_cases)}")
        print(f"{'*' * 70}\n")

        code, validation = test_pipeline(test_case)

        if validation.is_valid:
            print(f"\n✅ Test {i} PASSED - Code is valid!\n")
        else:
            print(f"\n❌ Test {i} FAILED - Code has errors!\n")

        input("\nPress Enter to continue to next test...\n")

    print("\n" + "🎉" * 35)
    print("ALL TESTS COMPLETE!")
    print("🎉" * 35 + "\n")