# Vis.py - Updated with improved colors, cleaner annotations, and better styling

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                           QGraphicsDropShadowEffect, QSizePolicy, QGridLayout, QPushButton)
from PyQt5.QtCore import Qt, pyqtSlot, QRectF, QEvent
from PyQt5.QtGui import QColor, QPainter, QFont, QBrush, QPen, QLinearGradient, QPicture, QIcon
import pyqtgraph as pg

class InfoPanel(QFrame):
    """Panel that displays current agent state and decision information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up styling with clean, grid-based layout
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            InfoPanel {
                background-color: rgba(240, 240, 240, 0.95);
                border-radius: 5px;
                border: 1px solid #aaa;
            }
            QLabel {
                font-family: 'Arial';
                padding: 1px;
                font-size: 10px;
                color: #333333;
            }
            .title {
                font-size: 12px;
                font-weight: bold;
                color: #333;
            }
            .value {
                font-size: 10px;
                color: #0066cc;
                font-weight: bold;
            }
        """)
        
        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout with reduced margins
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(2)  # Reduce spacing between elements
        
        # Title with reduced height
        self.title_label = QLabel("Current Agent State")
        self.title_label.setProperty("class", "title")
        self.layout.addWidget(self.title_label)
        
        # Add horizontal line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(line)
        
        # Create organized grid for metrics
        metrics_frame = QFrame()
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(4)  # Tight spacing
        
        # Row 0: Basic info in grid layout for better alignment
        metrics_layout.addWidget(QLabel("Action:"), 0, 0)
        self.action_value = QLabel("None")
        self.action_value.setProperty("class", "value")
        metrics_layout.addWidget(self.action_value, 0, 1)
        
        metrics_layout.addWidget(QLabel("Reward:"), 0, 2)
        self.reward_value = QLabel("0.0")
        self.reward_value.setProperty("class", "value")
        metrics_layout.addWidget(self.reward_value, 0, 3)
        
        metrics_layout.addWidget(QLabel("Cumulative:"), 0, 4)
        self.cumulative_value = QLabel("0.0")
        self.cumulative_value.setProperty("class", "value")
        metrics_layout.addWidget(self.cumulative_value, 0, 5)
        
        # Create state variables grid
        self.state_values = {}
        
        # Common state variables in Crafter
        common_vars = ['health', 'food', 'drink', 'energy', 'wood', 'stone']
        for i, var in enumerate(common_vars):
            row = i // 3 + 1  # 3 variables per row
            col = (i % 3) * 2  # Each variable takes 2 columns (label + value)
            
            metrics_layout.addWidget(QLabel(f"{var}:"), row, col)
            self.state_values[var] = QLabel("0.0")
            self.state_values[var].setProperty("class", "value")
            metrics_layout.addWidget(self.state_values[var], row, col + 1)
        
        self.layout.addWidget(metrics_frame)
        
        # Explanation section
        self.explanation_label = QLabel("Decision Explanation:")
        self.explanation_label.setProperty("class", "title")
        self.layout.addWidget(self.explanation_label)
        
        # Explanation text with reduced height
        self.explanation_text = QLabel("No decision data available.")
        self.explanation_text.setWordWrap(True)
        self.explanation_text.setMaximumHeight(40)  # Limit height
        self.layout.addWidget(self.explanation_text)
        
        # Set reduced heights
        self.setMinimumHeight(130)
        self.setMaximumHeight(150)  # Significantly reduced maximum height
    
    def update_state(self, step_data):
        """Update the display with new state information"""
        if not step_data:
            return
            
        # Update action
        action = step_data.get('action', 'None')
        self.action_value.setText(str(action))
        
        # Update reward
        reward = step_data.get('reward', 0.0)
        self.reward_value.setText(f"{reward:.2f}")
        
        # Update cumulative reward
        cumulative = step_data.get('cumulative_reward', 0.0)
        self.cumulative_value.setText(f"{cumulative:.2f}")
        
        # Create component_values dictionary
        component_values = {}
        
        # Update state variables
        for key, value in step_data.items():
            if key not in ['time_step', 'action', 'reward', 'cumulative_reward']:
                # Store non-zero values in component_values dictionary
                if value != 0:
                    component_values[key] = value
                    
                # Update UI if this state variable has a corresponding widget
                if key in self.state_values:
                    # Format based on value type
                    if isinstance(value, float):
                        self.state_values[key].setText(f"{value:.2f}")
                    else:
                        self.state_values[key].setText(f"{value}")
                    
                    # Colorize value
                    if value > 0:
                        self.state_values[key].setStyleSheet("color: #009900; font-weight: bold; font-size: 10px;")
                    elif value < 0:
                        self.state_values[key].setStyleSheet("color: #cc0000; font-weight: bold; font-size: 10px;")
                    else:
                        self.state_values[key].setStyleSheet("color: #0066cc; font-weight: bold; font-size: 10px;")
        
        # Generate explanation text based on the data
        self.generate_explanation(step_data, component_values)
    
    def generate_explanation(self, step_data, component_values):
        """Generate natural language explanation of the agent's decision"""
        action = step_data.get('action', 'None')
        reward = step_data.get('reward', 0.0)
        
        # Start with basic explanation
        explanation = f"The agent chose action '{action}'. "
        
        # No components case
        if not component_values:
            if reward == 0:
                explanation += "This action didn't affect the agent's state."
            elif reward > 0:
                explanation += f"This resulted in a positive reward of {reward:.2f}."
            else:
                explanation += f"This resulted in a negative reward of {reward:.2f}."
        else:
            # Sort components by absolute value
            sorted_components = sorted(component_values.items(), 
                                      key=lambda x: abs(x[1]), 
                                      reverse=True)
            
            # Identify the main component
            main_component, main_value = sorted_components[0]
            
            # Explain based on main component and value
            if main_value > 0:
                explanation += f"This primarily increased '{main_component}' by {main_value:.2f}, "
            elif main_value < 0:
                explanation += f"This primarily decreased '{main_component}' by {abs(main_value):.2f}, "
            
            # Add secondary components if present
            if len(sorted_components) > 1:
                secondary, sec_value = sorted_components[1]
                explanation += f"while also affecting '{secondary}' by {sec_value:.2f}. "
            else:
                explanation += "with no other significant effects. "
            
            # Overall assessment
            if reward > 0:
                explanation += f"Overall, this was a beneficial action with net reward {reward:.2f}."
            elif reward < 0:
                explanation += f"Overall, this was a detrimental action with net reward {reward:.2f}."
            else:
                explanation += "The net reward from this action was zero."
        
        self.explanation_text.setText(explanation)


class CustomBarGraphItem(pg.GraphicsObject):
    """Custom bar graph with improved styling and tooltips"""
    
    def __init__(self, x, height, width=0.8, brushes=None, pens=None):
        pg.GraphicsObject.__init__(self)
        self.x = np.array(x)
        self.height = np.array(height)
        self.width = width
        
        if brushes is None:
            self.brushes = [pg.mkBrush(100, 100, 255, 150) for _ in height]
        else:
            self.brushes = brushes
            
        if pens is None:
            self.pens = [pg.mkPen(None) for _ in height]
        else:
            self.pens = pens
            
        self._picture = None
        self._boundingRect = None
        self.generatePicture()
    
    def generatePicture(self):
        """Pre-render the bars as a QPicture object"""
        self._picture = QPicture()
        painter = QPainter(self._picture)
        
        for i in range(len(self.x)):
            x, h = self.x[i], self.height[i]
            
            if h == 0:  # Skip zero-height bars
                continue
                
            rect = QRectF(x - self.width/2, 0, self.width, h)
            painter.setBrush(self.brushes[i])
            painter.setPen(self.pens[i])
            
            # Draw rectangle with rounded corners for positive values
            if h > 0:
                painter.drawRoundedRect(rect, 2, 2)
            else:
                painter.drawRect(rect)
                
        painter.end()
        
        # Calculate bounding rect
        xmin = min(self.x) - self.width/2
        xmax = max(self.x) + self.width/2
        ymin = min(0, min(self.height))
        ymax = max(0, max(self.height))
        
        self._boundingRect = QRectF(xmin, ymin, xmax-xmin, ymax-ymin)
    
    def paint(self, painter, option, widget):
        painter.drawPicture(0, 0, self._picture)
    
    def boundingRect(self):
        return self._boundingRect


class DecisionPoint(pg.ScatterPlotItem):
    """Enhanced scatter plot item that highlights decision points"""
    
    def __init__(self, x, y, decision_type, importance, actions=None, **kwargs):
        self.decision_type = decision_type  # e.g., 'positive', 'negative', 'neutral'
        self.importance = importance  # Numeric importance (determines size)
        self.actions = actions if actions else []
        
        # Determine symbol based on decision type
        symbol = 'o'  # Default
        if decision_type == 'positive':
            symbol = 't'  # Triangle
        elif decision_type == 'negative':
            symbol = 'd'  # Diamond
        
        # Determine size based on importance
        size = 10 + (importance * 5)
        
        # Determine color based on decision type
        brush = pg.mkBrush(50, 50, 200, 200)  # Default blue
        if decision_type == 'positive':
            brush = pg.mkBrush(50, 200, 50, 200)  # Green
        elif decision_type == 'negative':
            brush = pg.mkBrush(200, 50, 50, 200)  # Red
        
        # Call parent constructor with calculated properties
        super().__init__(x=x, y=y, size=size, symbol=symbol, brush=brush, **kwargs)


class VisualizationWidget(QWidget):
    """Widget for displaying interactive visualizations of agent data"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize the UI
        self.init_ui()
        
        # Initialize data containers
        self.time_steps = []
        self.reward_log = []
        self.action_log = []
        self.reward_components = {}
        self.cumulative_rewards = []  # Store pre-calculated cumulative rewards
        
        # Variable to track current position
        self.current_step = 0
        
        # Set initial view state
        self.view_state = {
            'cumulative': True,
            'components': True
        }
        
        # Initialize tracking variables for hover items
        self.component_curves = {}
        self.cumulative_data_points = []
        self.hover_text = None
        self.components_hover_text = None
        self.highlight_point = None
        self.hover_line = None
    
    def init_ui(self):
        """Initialize UI components for visualization"""
        
        # Set up the layout
        layout = QVBoxLayout(self)
        layout.setSpacing(4)  # Reduced spacing between plots
        
        # Create info panel at top with lower height
        self.info_panel = InfoPanel()
        layout.addWidget(self.info_panel, 1)  # Lower stretch factor
        
        # Create a plot widget for the cumulative reward with more space
        self.cumulative_plot = pg.PlotWidget(title="Agent Reward Timeline")
        self.cumulative_plot.setBackground('w')  # White background
        self.cumulative_plot.setLabel('left', 'Reward')
        self.cumulative_plot.setLabel('bottom', 'Time Step')
        self.cumulative_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cumulative_plot.setMouseEnabled(x=True, y=True)
        
        # Change text color to dark for better visibility
        self.cumulative_plot.getAxis('left').setTextPen('k')  # Black text
        self.cumulative_plot.getAxis('bottom').setTextPen('k')  # Black text
        self.cumulative_plot.setTitle("Agent Reward Timeline", color="#333", size="12pt")
        
        # Install event filter to clear hover items when mouse leaves plot
        self.cumulative_plot.installEventFilter(self)
        
        # Style improvements for cumulative plot
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        axis_pen = pg.mkPen(color=(120, 120, 120), width=1)
        self.cumulative_plot.getAxis('bottom').setPen(axis_pen)
        self.cumulative_plot.getAxis('left').setPen(axis_pen)
        self.cumulative_plot.getAxis('bottom').setStyle(tickFont=font)
        self.cumulative_plot.getAxis('left').setStyle(tickFont=font)
        
        # Set higher stretch factor for plots
        layout.addWidget(self.cumulative_plot, 4)  # Higher stretch factor
        
        # Create a plot for reward components with more space
        self.components_plot = pg.PlotWidget(title="Reward Component Breakdown")
        self.components_plot.setBackground('w')  # White background
        self.components_plot.setLabel('left', 'Component Value')
        self.components_plot.setLabel('bottom', 'Time Step')
        self.components_plot.showGrid(x=True, y=True, alpha=0.3)
        self.components_plot.setMouseEnabled(x=True, y=True)
        
        # Change text color to dark for better visibility
        self.components_plot.getAxis('left').setTextPen('k')  # Black text
        self.components_plot.getAxis('bottom').setTextPen('k')  # Black text
        self.components_plot.setTitle("Reward Component Breakdown", color="#333", size="12pt")
        
        # Install event filter to clear hover items when mouse leaves plot
        self.components_plot.installEventFilter(self)
        
        # Style improvements for components plot
        self.components_plot.getAxis('bottom').setPen(axis_pen)
        self.components_plot.getAxis('left').setPen(axis_pen)
        self.components_plot.getAxis('bottom').setStyle(tickFont=font)
        self.components_plot.getAxis('left').setStyle(tickFont=font)
        
        # Set higher stretch factor for plots
        layout.addWidget(self.components_plot, 4)  # Higher stretch factor
        
        # Create position markers (vertical lines)
        self.cumulative_position_line = pg.InfiniteLine(
            angle=90, 
            movable=False, 
            pen=pg.mkPen('r', width=2, style=Qt.DashLine)
        )
        self.components_position_line = pg.InfiniteLine(
            angle=90, 
            movable=False, 
            pen=pg.mkPen('r', width=2, style=Qt.DashLine)
        )
        
        self.cumulative_plot.addItem(self.cumulative_position_line)
        self.components_plot.addItem(self.components_position_line)
        
        # Add legend to components plot
        self.components_legend = self.components_plot.addLegend(offset=(-10, 10))
        
        # Set up proxy for mouse hover events
        self.cumulative_proxy = pg.SignalProxy(
            self.cumulative_plot.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.on_cumulative_hover
        )
        
        self.components_proxy = pg.SignalProxy(
            self.components_plot.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.on_components_hover
        )
        
        # Add highlighted step region for context
        self.highlighted_region = pg.LinearRegionItem(
            values=[0, 0],
            brush=pg.mkBrush(100, 100, 255, 20),
            pen=pg.mkPen(None),
            movable=False
        )
        self.cumulative_plot.addItem(self.highlighted_region)
        self.highlighted_region.setVisible(False)
        
        # Add current step annotation
        self.current_step_text = pg.TextItem(
            text="",
            color=(200, 0, 0),
            anchor=(0.5, 0)
        )
        self.cumulative_plot.addItem(self.current_step_text)
        
        # Set plot size policies to allow expansion
        self.cumulative_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.components_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def eventFilter(self, obj, event):
        """Event filter to clear hover items when mouse leaves plot area"""
        if event.type() == QEvent.Leave:
            if obj == self.cumulative_plot:
                self.clear_cumulative_hover_items()
            elif obj == self.components_plot:
                self.clear_components_hover_items()
        return super().eventFilter(obj, event)
    
    def clear_cumulative_hover_items(self):
        """Clear all hover items from cumulative plot"""
        if hasattr(self, 'hover_text') and self.hover_text is not None:
            try:
                self.cumulative_plot.removeItem(self.hover_text)
            except:
                pass
        
        if hasattr(self, 'highlight_point') and self.highlight_point is not None:
            try:
                self.cumulative_plot.removeItem(self.highlight_point)
            except:
                pass
    
    def clear_components_hover_items(self):
        """Clear all hover items from components plot"""
        if hasattr(self, 'components_hover_text') and self.components_hover_text is not None:
            try:
                self.components_plot.removeItem(self.components_hover_text)
            except:
                pass
        
        if hasattr(self, 'hover_line') and self.hover_line is not None:
            try:
                self.components_plot.removeItem(self.hover_line)
            except:
                pass
    
    def set_data(self, time_steps, reward_log, action_log, reward_components):
        """Set the data for visualization"""
        
        # Store the data
        self.time_steps = time_steps
        self.reward_log = reward_log
        self.action_log = action_log
        self.reward_components = reward_components
        
        # Pre-calculate cumulative rewards
        self.cumulative_rewards = np.cumsum(self.reward_log)
        
        # Clear existing plots
        self.cumulative_plot.clear()
        self.components_plot.clear()
        
        # Re-add position markers
        self.cumulative_plot.addItem(self.cumulative_position_line)
        self.components_plot.addItem(self.components_position_line)
        self.cumulative_plot.addItem(self.current_step_text)
        self.cumulative_plot.addItem(self.highlighted_region)
        
        # Reset the components legend
        if hasattr(self, 'components_legend') and self.components_legend:
            if hasattr(self.components_legend, 'scene') and callable(self.components_legend.scene) and self.components_legend.scene():
                self.components_legend.scene().removeItem(self.components_legend)
        self.components_legend = self.components_plot.addLegend(offset=(-10, 10))
        
        # Update the plots with new data
        self.update_cumulative_plot()
        self.update_components_plot()
        
        # Initialize position at step 0
        self.update_position(0)
        
        # Store data points for hover lookup
        self.cumulative_data_points = []
        for i, (t, r, a, c) in enumerate(zip(self.time_steps, self.reward_log, 
                                           self.action_log, self.cumulative_rewards)):
            self.cumulative_data_points.append({
                'x': t,
                'y': c,
                'time_step': t,
                'action': a,
                'reward': r,
                'cumulative': c,
                'index': i
            })
        
        # Calculate visualization ranges and adjust axes
        if time_steps:
            x_min, x_max = min(time_steps), max(time_steps)
            x_padding = (x_max - x_min) * 0.05  # 5% padding
            
            # Find y-ranges with padding
            if self.cumulative_rewards.size > 0:
                y_min = min(0, np.min(self.cumulative_rewards))
                y_max = max(0, np.max(self.cumulative_rewards))
                y_padding = max((y_max - y_min) * 0.1, 0.5)  # 10% padding or at least 0.5
                
                # Set cumulative plot range
                self.cumulative_plot.setXRange(x_min - x_padding, x_max + x_padding)
                self.cumulative_plot.setYRange(y_min - y_padding, y_max + y_padding)
            
            # Set component plot range
            self.components_plot.setXRange(x_min - x_padding, x_max + x_padding)
    
    def update_cumulative_plot(self):
        """Update the cumulative reward plot with interactive features"""
        
        if not self.time_steps or not self.reward_log:
            return
        
        # Add a legend for decision points with dark text
        legend = self.cumulative_plot.addLegend(offset=(10, 10), loc = 'top-left')
        legend.addItem(pg.ScatterPlotItem(symbol='t', brush=(50, 200, 50, 200)), "Positive Reward")
        legend.addItem(pg.ScatterPlotItem(symbol='d', brush=(200, 50, 50, 200)), "Negative Reward")
        legend.addItem(pg.PlotDataItem(pen=pg.mkPen(color=(0, 0, 255), width=2)), "Cumulative Reward")
        
        # Apply dark text to legend
        for item in legend.items:
            item[1].setText(item[1].text, color='#333')
        
        # Create main line plot
        pen = pg.mkPen(color=(0, 0, 255), width=2.5)
        self.cumulative_curve = self.cumulative_plot.plot(
            self.time_steps, 
            self.cumulative_rewards, 
            pen=pen, 
            name=None
        )
        
        # Identify reward change points (non-zero rewards)
        non_zero_indices = [i for i, r in enumerate(self.reward_log) if abs(r) > 0.001]
        
        if non_zero_indices:
            # Create improved scatter points for reward changes
            self.reward_points = []
            
            for i in non_zero_indices:
                # Determine point characteristics based on reward
                reward = self.reward_log[i]
                cum_reward = self.cumulative_rewards[i]
                
                # Classify by reward impact
                if reward > 0:
                    decision_type = 'positive'
                    importance = min(reward / 0.5, 2.0)  # Scale importance, max at 2.0
                elif reward < 0:
                    decision_type = 'negative'
                    importance = min(abs(reward) / 0.5, 2.0)
                else:
                    decision_type = 'neutral'
                    importance = 0.5
                
                # Create decision point
                point = DecisionPoint(
                    x=[self.time_steps[i]], 
                    y=[cum_reward],
                    decision_type=decision_type,
                    importance=importance,
                    actions=[self.action_log[i]]
                )
                self.cumulative_plot.addItem(point)
                self.reward_points.append(point)
        
        # Add step rewards as a bar graph with custom styling
        if self.reward_log:
            # Create custom brushes based on reward values
            brushes = []
            pens = []
            
            for r in self.reward_log:
                if r > 0:
                    # Positive reward - green gradient
                    brushes.append(pg.mkBrush(100, min(100 + r*100, 255), 100, 150))
                    pens.append(pg.mkPen(0, 150, 0, 100, width=0.5))
                elif r < 0:
                    # Negative reward - red gradient
                    brushes.append(pg.mkBrush(min(100 + abs(r)*100, 255), 100, 100, 150))
                    pens.append(pg.mkPen(150, 0, 0, 100, width=0.5))
                else:
                    # Zero reward - grey
                    brushes.append(pg.mkBrush(150, 150, 150, 50))
                    pens.append(pg.mkPen(None))
            
            # Create and add custom bar graph
            try:
                reward_bars = CustomBarGraphItem(
                    x=self.time_steps,
                    height=self.reward_log,
                    width=0.6,
                    brushes=brushes,
                    pens=pens
                )
                self.cumulative_plot.addItem(reward_bars)
            except Exception as e:
                print(f"Error creating custom bar graph: {e}")
            
        # Add zero line
        zero_line = pg.InfiniteLine(
            pos=0, 
            angle=0, 
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DashLine)
        )
        self.cumulative_plot.addItem(zero_line)
    
    def update_components_plot(self):
        """Update the reward components plot with interactive features"""
        
        if not self.time_steps or not self.reward_components:
            return
        
        # Create a colormap with more vibrant, distinguishable colors
        colors = [
            (255, 50, 50),     # Bright Red
            (0, 180, 0),       # Bright Green
            (50, 100, 255),    # Bright Blue
            (255, 200, 0),     # Bright Yellow
            (200, 0, 200),     # Bright Magenta
            (0, 200, 200),     # Bright Cyan
            (255, 100, 0),     # Bright Orange
            (150, 50, 250),    # Bright Purple
            (0, 100, 100),     # Teal
            (180, 0, 100)      # Crimson
        ]
        
        # Store curves for hover functionality
        self.component_curves = {}
        
        # Limit to components with non-zero values
        active_components = {}
        for key, values in self.reward_components.items():
            if any(v != 0 for v in values):
                active_components[key] = values
        
        # Track area items for cleanup
        self.component_areas = []
        
        # Calculate baseline positions for stacked areas
        baseline = np.zeros(len(self.time_steps))
        positive_base = np.zeros(len(self.time_steps))
        negative_base = np.zeros(len(self.time_steps))
        
        # Get sorted components for stacking (always positive components on top)
        components_max = {name: max(abs(min(values)), abs(max(values))) 
                         for name, values in active_components.items()}
        sorted_components = sorted(active_components.items(), 
                                  key=lambda x: components_max[x[0]], 
                                  reverse=True)
        
        # For each component, create a filled area plot
        for i, (name, values) in enumerate(sorted_components):
            # Select color
            color = colors[i % len(colors)]
            
            # Create pens with gradient for fill
            pen = pg.mkPen(color=color, width=2)
            
            # Split into positive and negative
            values_array = np.array(values)
            pos_values = np.copy(values_array)
            pos_values[pos_values < 0] = 0
            
            neg_values = np.copy(values_array)
            neg_values[neg_values > 0] = 0
            
            # Create gradient fill for positive values
            if np.any(pos_values > 0):
                fill_brush = pg.mkBrush(color[0], color[1], color[2], 80)  # More opacity
                
                # Plot as stacked area
                fill_curve = pg.FillBetweenItem(
                    pg.PlotDataItem(self.time_steps, positive_base + pos_values), 
                    pg.PlotDataItem(self.time_steps, positive_base), 
                    brush=fill_brush
                )
                self.components_plot.addItem(fill_curve)
                self.component_areas.append(fill_curve)
                
                # Update the baseline for next component
                positive_base = positive_base + pos_values
            
            # Create gradient fill for negative values 
            if np.any(neg_values < 0):
                fill_brush = pg.mkBrush(color[0], color[1], color[2], 80)  # More opacity
                
                # Plot as stacked area
                fill_curve = pg.FillBetweenItem(
                    pg.PlotDataItem(self.time_steps, negative_base), 
                    pg.PlotDataItem(self.time_steps, negative_base + neg_values), 
                    brush=fill_brush
                )
                self.components_plot.addItem(fill_curve)
                self.component_areas.append(fill_curve)
                
                # Update the baseline for next component
                negative_base = negative_base + neg_values
            
            # Draw the line on top
            curve = self.components_plot.plot(
                self.time_steps, 
                values, 
                pen=pen, 
                name=name
            )
            
            self.component_curves[name] = {
                'curve': curve,
                'values': values,
                'color': color
            }
        
        # Add zero line
        zero_line = pg.InfiniteLine(
            pos=0, 
            angle=0, 
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DashLine)
        )
        self.components_plot.addItem(zero_line)
    
    def update_position(self, step, from_video=False, from_timeline=False):
        """Update the position marker in visualizations"""
        
        if not self.time_steps or step >= len(self.time_steps):
            return
        
        # Update current step
        self.current_step = step
        
        # Get the x-position (time step)
        x_pos = self.time_steps[step]
        
        # Update position lines
        self.cumulative_position_line.setValue(x_pos)
        self.components_position_line.setValue(x_pos)
        
        # Update highlighted region for context (5 steps before and after)
        context_start = max(0, step - 5)
        context_end = min(len(self.time_steps) - 1, step + 5)
        if context_start < context_end:
            self.highlighted_region.setRegion([
                self.time_steps[context_start], 
                self.time_steps[context_end]
            ])
            self.highlighted_region.setVisible(True)
        else:
            self.highlighted_region.setVisible(False)
        
        # Update step annotation (without Step/Action/Reward labels)
        if self.action_log and step < len(self.action_log):
            action = self.action_log[step]
            
            # Create minimal marker for the current position
            self.current_step_text.setHtml(
                f"<div style='background-color: rgba(255, 255, 255, 0.8); padding: 2px 5px; "
                f"border: 1px solid #aaa; font-size: 9px;'>"
                f"<span style='color: #000;'>{x_pos}</span>"
                f"</div>"
            )
            self.current_step_text.setPos(x_pos, self.cumulative_rewards[step])
            
            # Update info panel with detailed state information
            if hasattr(self, 'info_panel'):
                # Gather all data for this step
                step_data = {
                    'time_step': x_pos,
                    'action': action,
                    'reward': self.reward_log[step],
                    'cumulative_reward': self.cumulative_rewards[step]
                }
                
                # Add component values
                for name, values in self.reward_components.items():
                    if step < len(values):
                        step_data[name] = values[step]
                
                self.info_panel.update_state(step_data)
    
    def on_cumulative_hover(self, event):
        """Handle hover events on the cumulative plot with compact tooltips"""
        
        # Check if data is available
        if not hasattr(self, 'cumulative_data_points') or not self.cumulative_data_points:
            return
            
        # Convert the event position to plot coordinates
        pos = event[0]
        if not self.cumulative_plot.sceneBoundingRect().contains(pos):
            return
            
        mouse_point = self.cumulative_plot.getViewBox().mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Find the closest data point
        closest_point = min(self.cumulative_data_points, 
                           key=lambda p: abs(p['x'] - x))
        
        # Only update if we're close enough to the point
        if abs(closest_point['x'] - x) > (max(self.time_steps) - min(self.time_steps)) / 20:
            return
        
        # Clear previous hover items
        self.clear_cumulative_hover_items()
        
        # Create a temporary point to highlight hover position
        self.highlight_point = pg.ScatterPlotItem(
            [closest_point['x']], 
            [closest_point['y']], 
            size=12, 
            brush=pg.mkBrush(255, 255, 0, 200),
            pen=pg.mkPen(255, 255, 0, 200, width=2)
        )
        self.cumulative_plot.addItem(self.highlight_point)
        
        # Create tooltip at hover position with compact sizing
        self.hover_text = pg.TextItem(
            anchor=(0, 0),
            border=pg.mkPen((50, 50, 50, 100), width=1),
            fill=pg.mkBrush(255, 255, 255, 230)
        )
        
        # Format tooltip text with smaller font and dark text color
        tooltip_html = (
            f"<span style='color: #333; font-size: 9px;'>"
            f"<b>Step:</b> {closest_point['time_step']}<br>"
            f"<b>Action:</b> {closest_point['action']}<br>"
            f"<b>Reward:</b> {closest_point['reward']:.2f}<br>"
            f"<b>Cumulative:</b> {closest_point['cumulative']:.2f}"
            f"</span>"
        )
        
        self.hover_text.setHtml(tooltip_html)
        
        # Position tooltip to remain visible within view
        view_rect = self.cumulative_plot.viewRect()
        x_pos = closest_point['x'] + 1
        y_pos = closest_point['y']
        
        # Adjust position if tooltip would go outside view
        if x_pos + 100 > view_rect.right():  # Assuming tooltip width ~100px
            x_pos = closest_point['x'] - 1
            self.hover_text.setAnchor((1, 0))  # Right-aligned
        
        self.hover_text.setPos(x_pos, y_pos)
        self.cumulative_plot.addItem(self.hover_text)
    
    def on_components_hover(self, event):
        """Handle hover events on the components plot with compact tooltips"""
        
        # Check if component_curves exists
        if not hasattr(self, 'component_curves') or not self.component_curves:
            return
            
        # Convert the event position to plot coordinates
        pos = event[0]
        if not self.components_plot.sceneBoundingRect().contains(pos):
            return
            
        mouse_point = self.components_plot.getViewBox().mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Find the closest time step
        if not self.time_steps:
            return
            
        closest_idx = min(range(len(self.time_steps)), 
                         key=lambda i: abs(self.time_steps[i] - x))
            
        # Only update if we're close enough to a time step
        if abs(self.time_steps[closest_idx] - x) > (max(self.time_steps) - min(self.time_steps)) / 20:
            return
        
        # Clear previous hover items
        self.clear_components_hover_items()
        
        # Gather component values at this time step
        component_values = {}
        for name, data in self.component_curves.items():
            values = data['values']
            if closest_idx < len(values):
                component_values[name] = values[closest_idx]
        
        # Create hover tooltip with compact sizing
        self.components_hover_text = pg.TextItem(
            anchor=(0, 0),
            border=pg.mkPen((50, 50, 50, 100), width=1),
            fill=pg.mkBrush(255, 255, 255, 230)
        )
        
        # Format tooltip HTML with smaller font and dark text color
        tooltip_html = (
            f"<span style='color: #333; font-size: 9px;'>"
            f"<b>Step:</b> {self.time_steps[closest_idx]}<br>"
            f"<b>Action:</b> {self.action_log[closest_idx] if closest_idx < len(self.action_log) else ''}<br>"
        )
        
        # Add component values (limited to top 5 by value)
        sorted_components = sorted(component_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for name, value in sorted_components:
            color = self.component_curves[name]['color']
            tooltip_html += (
                f"<span style='color: rgb({color[0]},{color[1]},{color[2]});'>"
                f"<b>{name}:</b> {value:.2f}</span><br>"
            )
        
        tooltip_html += "</span>"
        
        self.components_hover_text.setHtml(tooltip_html)
        
        # Position tooltip to remain visible within view
        view_rect = self.components_plot.viewRect()
        x_pos = self.time_steps[closest_idx] + 1
        
        # Adjust position if tooltip would go outside view
        if x_pos + 100 > view_rect.right():  # Assuming tooltip width ~100px
            x_pos = self.time_steps[closest_idx] - 1
            self.components_hover_text.setAnchor((1, 0))  # Right-aligned
        
        self.components_hover_text.setPos(x_pos, y)
        self.components_plot.addItem(self.components_hover_text)
        
        # Create a temporary vertical line at hover position
        self.hover_line = pg.InfiniteLine(
            pos=self.time_steps[closest_idx], 
            angle=90, 
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DotLine)
        )
        self.components_plot.addItem(self.hover_line)
    
    def toggle_view(self, view_type, visible):
        """Toggle visibility of different visualization types"""
        
        self.view_state[view_type] = visible
        
        # Update visibility
        if view_type == 'cumulative':
            self.cumulative_plot.setVisible(visible)
        elif view_type == 'components':
            self.components_plot.setVisible(visible)
