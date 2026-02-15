# backend/code_generator.py
"""
Code Generator - Converts UI plans into React code
Pure template-based generation, no LLM needed
"""

from typing import Dict, List, Any
import json

# --- BEGIN INLINED planner.py content ---
# This content was moved here to resolve ModuleNotFoundError
# as direct imports between Colab cells posing as files is not automatic.

# --- BEGIN INLINED intent_parser.py content ---
# This content was moved here to resolve ModuleNotFoundError
# as direct imports between Colab cells posing as files is not automatic.
import re
# from typing import Dict, List, Optional, Any # Already imported Dict, List, Any above
from dataclasses import dataclass # Already imported dataclass above

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

    # Keyword mappings
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
        """Initialize the intent parser"""
        pass

    def parse(self, user_input: str) -> Intent:
        """
        Parse user input and return intent

        Args:
            user_input: Raw user input string

        Returns:
            Intent object with parsed information
        """
        user_input = user_input.lower().strip()

        # Extract action
        action = self._extract_action(user_input)

        # Extract UI type
        ui_type = self._extract_ui_type(user_input)

        # Extract components
        components = self._extract_components(user_input)

        # Extract layout
        layout = self._extract_layout(user_input)

        # Extract modifiers (variant, color, etc.)
        modifiers = self._extract_modifiers(user_input)

        # Calculate confidence
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
        """Extract the primary action from text"""
        for action, keywords in self.ACTIONS.items():
            if any(keyword in text for keyword in keywords):
                return action
        return 'create'  # Default action

    def _extract_ui_type(self, text: str) -> str:
        """Extract the UI type from text"""
        for ui_type, keywords in self.UI_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return ui_type

        # If no specific type found, infer from components
        if 'input' in text and 'button' in text:
            return 'form'
        elif 'card' in text or 'kpi' in text:
            return 'dashboard'

        return 'generic'  # Default

    def _extract_components(self, text: str) -> List[str]:
        """Extract component names from text"""
        found_components = []

        for component, keywords in self.COMPONENTS.items():
            if any(keyword in text for keyword in keywords):
                if component not in found_components:
                    found_components.append(component)

        # If no components found, infer from UI type
        if not found_components:
            found_components = self._infer_components_from_ui_type(
                self._extract_ui_type(text)
            )

        return found_components

    def _extract_layout(self, text: str) -> str:
        """Extract layout type from text"""
        for layout, keywords in self.LAYOUTS.items():
            if any(keyword in text for keyword in keywords):
                return layout
        return 'flex'  # Default layout

    def _extract_modifiers(self, text: str) -> Dict[str, any]:
        """Extract additional modifiers (variant, color, etc.)"""
        modifiers = {}

        # Extract variant
        for variant, keywords in self.VARIANTS.items():
            if any(keyword in text for keyword in keywords):
                modifiers['variant'] = variant
                break

        # Extract color
        for color, keywords in self.COLORS.items():
            if any(keyword in text for keyword in keywords):
                modifiers['color'] = color
                break

        # Extract size
        if 'large' in text or 'big' in text:
            modifiers['size'] = 'large'
        elif 'small' in text or 'tiny' in text:
            modifiers['size'] = 'small'

        # Extract numbers (for tables, charts, etc.)
        numbers = re.findall(r'\\b(\\d+)\\b', text)
        if numbers:
            modifiers['count'] = int(numbers[0])

        return modifiers

    def _infer_components_from_ui_type(self, ui_type: str) -> List[str]:
        """Infer default components based on UI type"""
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
        """Calculate confidence score for the parsed intent"""
        score = 0.5  # Base score

        if action != 'create':
            score += 0.1

        if ui_type != 'generic':
            score += 0.2

        if components:
            score += 0.2

        return min(score, 1.0)
# --- END INLINED intent_parser.py content ---

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

    # Template definitions for common UI patterns
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
        """Initialize the planner"""
        pass

    def create_plan(self, intent: Intent) -> UIPlan:
        """
        Create a UI plan from intent

        Args:
            intent: Parsed intent object

        Returns:
            UIPlan with complete component structure
        """
        # Get template based on UI type
        template = self.TEMPLATES.get(intent.ui_type, self.TEMPLATES['form'])

        # Create container props
        container_props = template['container'].copy()

        # Plan components
        components = self._plan_components(intent, template)

        # Determine imports
        imports = self._determine_imports(components)

        # Generate reasoning
        reasoning = self._generate_reasoning(intent, components)

        return UIPlan(
            layout_type=template['layout'],
            container_props=container_props,
            components=components,
            imports=imports,
            reasoning=reasoning
        )

    def _plan_components(self, intent: Intent, template: Dict) -> List[ComponentPlan]:
        """Plan individual components based on intent"""
        planned_components = []

        if intent.components:
            # User specified components
            for idx, component_type in enumerate(intent.components):
                component = self._create_component_plan(
                    component_type,
                    intent.modifiers,
                    position={'row': idx // 3, 'col': idx % 3}
                )
                planned_components.append(component)
        else:
            # Use template defaults
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
        """Create a plan for a single component"""

        # Base props for each component type
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

        # Apply modifiers
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
        """Determine required imports based on components"""
        component_types = set(comp.type for comp in components)

        imports = [
            f"import {comp_type} from '@/components/ui/{comp_type}';"
            for comp_type in component_types
        ]

        # Add recharts import if Chart is used
        if 'Chart' in component_types:
            imports.append("import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';")

        return imports

    def _generate_reasoning(self, intent: Intent,
                           components: List[ComponentPlan]) -> str:
        """Generate human-readable reasoning for the plan"""
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

class CodeGenerator:
    """Generates React code from UI plans"""

    def __init__(self):
        """Initialize the code generator"""
        pass

    def generate(self, plan: UIPlan) -> str:
        """
        Generate React code from a UI plan

        Args:
            plan: UIPlan object with component structure

        Returns:
            Complete React component code as string
        """
        # Generate imports
        imports = self._generate_imports(plan.imports)

        # Generate component code
        component_code = self._generate_component(plan)

        # Combine into complete code
        full_code = f"{imports}\n\n{component_code}"

        return full_code

    def _generate_imports(self, imports: List[str]) -> str:
        """Generate import statements"""
        if not imports:
            return ""

        return "\n".join(imports)

    def _generate_component(self, plan: UIPlan) -> str:
        """Generate the main component code"""
        # Generate JSX for all components
        components_jsx = []
        for comp_plan in plan.components:
            jsx = self._generate_component_jsx(comp_plan)
            components_jsx.append(jsx)

        # Get container classes
        container_class = plan.container_props.get('className', '')

        # Build component
        code = f"""export default function GeneratedComponent() {{
  return (
    <div className="{container_class}">
{self._indent(chr(10).join(components_jsx), 6)}
    </div>
  );
}}"""

        return code

    def _generate_component_jsx(self, comp_plan: ComponentPlan) -> str:
        """Generate JSX for a single component"""
        comp_type = comp_plan.type
        props = comp_plan.props

        # Component-specific JSX generation
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
        """Generate Button JSX"""
        variant = props.get('variant', 'primary')
        children = props.get('children', 'Button')

        return f'<Button variant="{variant}">{children}</Button>'

    def _generate_card(self, props: Dict) -> str:
        """Generate Card JSX"""
        title = props.get('title', 'Card Title')
        content = props.get('content', 'Card content goes here.')

        return f'''<Card>
  <Card.Title>{title}</Card.Title>
  <Card.Content>
    <p>{content}</p>
  </Card.Content>
</Card>'''

    def _generate_input(self, props: Dict) -> str:
        """Generate Input JSX"""
        label = props.get('label', 'Label')
        placeholder = props.get('placeholder', 'Enter value...')

        return f'<Input label="{label}" placeholder="{placeholder}" />'

    def _generate_table(self, props: Dict) -> str:
        """Generate Table JSX"""
        columns = props.get('columns', ['Column 1', 'Column 2'])
        data = props.get('data', [])

        # Generate column definitions
        col_defs = ', '.join([f'{{ header: "{col}", accessor: "{col.lower().replace(" ", "_")}" }}'
                             for col in columns])

        # Generate data array
        data_str = json.dumps(data, indent=2)

        return f'''<Table>
  columns={{[{col_defs}]}}
  data={{data}}
/>'''

    def _generate_chart(self, props: Dict) -> str:
        """Generate Chart JSX"""
        chart_type = props.get('type', 'line')
        data = props.get('data', [])

        data_str = json.dumps(data, indent=2)

        if chart_type == 'line':
            return f'''<Chart type="line" data={{{data_str}}} />'''
        elif chart_type == 'bar':
            return f'''<Chart type="bar" data={{{data_str}}} />'''
        else:
            return f'''<Chart type="line" data={{{data_str}}} />'''

    def _generate_navbar(self, props: Dict) -> str:
        """Generate Navbar JSX"""
        brand = props.get('brand', 'Brand')
        links = props.get('links', ['Home', 'About'])

        return f'''<Navbar brand="{brand}">
  {' '.join([f'<Navbar.Link>{link}</Navbar.Link>' for link in links])}
</Navbar>'''

    def _generate_sidebar(self, props: Dict) -> str:
        """Generate Sidebar JSX"""
        items = props.get('items', ['Item 1', 'Item 2'])

        items_jsx = '\n  '.join([f'<Sidebar.Item>{item}</Sidebar.Item>' for item in items])

        return f'''<Sidebar>
  {items_jsx}
</Sidebar>'''

    def _generate_modal(self, props: Dict) -> str:
        """Generate Modal JSX"""
        title = props.get('title', 'Modal Title')
        children = props.get('children', 'Modal content')

        return f'''<Modal>
  <Modal.Title>{title}</Modal.Title>
  <Modal.Content>
    <p>{children}</p>
  </Modal.Content>
  <Modal.Footer>
    <Button variant="primary">Save</Button>
    <Button variant="secondary">Cancel</Button>
  </Modal.Footer>
</Modal>'''

    def _indent(self, text: str, spaces: int) -> str:
        """Add indentation to text"""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line
                        for line in text.split('\n'))


# Example usage
if __name__ == '__main__':

    parser = IntentParser()
    planner = Planner()
    generator = CodeGenerator()

    test_input = "Create a dashboard with 2 cards and a chart"

    # Pipeline
    intent = parser.parse(test_input)
    plan = planner.create_plan(intent)
    code = generator.generate(plan)

    print(f"Input: {test_input}\n")
    print("Generated Code:")
    print("=" * 60)
    print(code)
    print("=" * 60)