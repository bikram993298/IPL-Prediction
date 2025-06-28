# IPL Win Probability Predictor

A sophisticated real-time cricket analytics application that predicts IPL match win probabilities using advanced machine learning algorithms and live cricket data integration.

![IPL Win Predictor](https://images.pexels.com/photos/1661950/pexels-photo-1661950.jpeg?auto=compress&cs=tinysrgb&w=1200&h=400&fit=crop)

## 🏏 Overview

This application combines cutting-edge machine learning with real-time cricket data to provide accurate win probability predictions for IPL matches. Built with React, TypeScript, and Tailwind CSS, it offers a beautiful, production-ready interface with live data integration capabilities.

## ✨ Key Features

### 🎯 Core Functionality
- **Real-time Win Probability Calculation** - Advanced ML algorithms for accurate predictions
- **Interactive Team Selection** - Choose from all 8 IPL teams with authentic branding
- **Dynamic Match Scenarios** - Support for both first and second innings
- **Comprehensive Match Factors** - Momentum, pressure, form, and conditions analysis
- **Beautiful Visualizations** - Real-time probability charts and match progression

### 📡 Live Data Integration
- **Real-time Score Updates** - Connect to live IPL matches via multiple APIs
- **Auto-sync Match Data** - Automatically populate all prediction inputs
- **Multiple Data Sources** - CricAPI, RapidAPI, and fallback systems
- **Live Match Selection** - Choose from currently active IPL matches
- **30-second Updates** - Real-time polling during live matches

### 🎨 User Experience
- **Apple-level Design** - Premium UI with attention to detail
- **Responsive Layout** - Optimized for all devices and screen sizes
- **Smooth Animations** - Micro-interactions and hover states
- **Accessibility** - WCAG compliant with proper contrast ratios
- **Performance Optimized** - Fast loading and smooth interactions

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Modern web browser
- (Optional) Cricket API keys for live data

### Installation

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd ipl-win-predictor
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Open in Browser**
   ```
   http://localhost:5173
   ```

## 📊 How It Works

### Prediction Algorithm

The win probability calculation considers multiple factors:

```typescript
// Core factors influencing predictions
- Current match situation (score, wickets, overs)
- Required run rate vs current run rate
- Wickets in hand and pressure situations
- Team strengths and historical performance
- Venue advantages and toss decisions
- Weather conditions and match phase
```

### Real-time Integration

```typescript
// Live data flow
1. Fetch live matches from cricket APIs
2. Select active IPL match
3. Auto-populate match parameters
4. Calculate real-time win probabilities
5. Update every 30 seconds during live play
```

## 🔧 Configuration

### Live Data Setup

#### Option 1: CricAPI (Recommended)
```javascript
// src/services/cricketApi.ts
const API_KEY = 'your-cricapi-key';
```

1. Sign up at [cricapi.com](https://cricapi.com)
2. Get your free API key (100 requests/day)
3. Replace `'YOUR_API_KEY'` in the code

#### Option 2: RapidAPI Cricket Live Data
```javascript
// src/services/cricketApi.ts
const RAPIDAPI_KEY = 'your-rapidapi-key';
```

1. Subscribe at [RapidAPI Cricket Live Data](https://rapidapi.com/cricketapi/api/cricket-live-data)
2. Get your RapidAPI key
3. Replace `'YOUR_RAPIDAPI_KEY'` in the code

#### Option 3: Mock Data (Development)
No setup required - works out of the box with realistic simulated data.

## 🏗️ Project Structure

```
src/
├── components/           # React components
│   ├── TeamSelector.tsx     # Team selection interface
│   ├── MatchDetails.tsx     # Match configuration
│   ├── ScoreInput.tsx       # Score and overs input
│   ├── ProbabilityDisplay.tsx # Win probability visualization
│   ├── MatchFactors.tsx     # Key factors analysis
│   ├── LiveMatchSelector.tsx # Live match selection
│   └── LiveMatchIntegration.tsx # Live data integration
├── hooks/               # Custom React hooks
│   ├── useProbabilityCalculator.ts # ML prediction logic
│   └── useLiveMatches.ts    # Live data management
├── services/            # External services
│   └── cricketApi.ts        # Cricket API integration
└── App.tsx             # Main application component
```

## 🎯 Usage Examples

### Basic Prediction
```typescript
// Set up a match scenario
const matchState = {
  team1: 'Chennai Super Kings',
  team2: 'Mumbai Indians',
  currentScore: 120,
  wickets: 3,
  overs: 15,
  target: 180,
  venue: 'Wankhede Stadium, Mumbai'
};

// Get win probability
const probability = calculateWinProbability(matchState);
// Returns: { team1: 65.2, team2: 34.8, confidence: 'high' }
```

### Live Data Integration
```typescript
// Connect to live match
const liveMatch = await cricketApi.getLiveMatches();
const selectedMatch = liveMatch[0];

// Auto-populate match data
updateMatchState(selectedMatch);

// Real-time updates every 30 seconds
setInterval(() => {
  updateLiveData(selectedMatch.matchId);
}, 30000);
```

## 🔬 Technical Details

### Machine Learning Features

The prediction algorithm incorporates:

- **Pressure Indicators**: Run rate pressure, wicket pressure, required rate analysis
- **Momentum Scores**: Current form, scoring rate, wickets in hand
- **Match Context**: Venue advantage, toss impact, weather conditions
- **Phase Analysis**: Powerplay, middle overs, death overs performance
- **Historical Data**: Team head-to-head records and recent form

### Performance Optimizations

- **React.memo** for component optimization
- **useMemo** and **useCallback** for expensive calculations
- **Debounced updates** for real-time data
- **Lazy loading** for non-critical components
- **Optimized bundle size** with tree shaking

## 🎨 Design System

### Color Palette
```css
/* Team Colors */
--csk-primary: #FBBF24;    /* Chennai Super Kings */
--mi-primary: #2563EB;     /* Mumbai Indians */
--rcb-primary: #DC2626;    /* Royal Challengers Bangalore */
--kkr-primary: #7C3AED;    /* Kolkata Knight Riders */

/* UI Colors */
--primary: #3B82F6;
--secondary: #6B7280;
--success: #10B981;
--warning: #F59E0B;
--error: #EF4444;
```

### Typography
```css
/* Font System */
--font-heading: 'Inter', sans-serif;
--font-body: 'Inter', sans-serif;

/* Font Weights */
--weight-normal: 400;
--weight-medium: 500;
--weight-semibold: 600;
--weight-bold: 700;
```

## 📱 Responsive Design

### Breakpoints
```css
/* Mobile First Approach */
sm: 640px   /* Small devices */
md: 768px   /* Medium devices */
lg: 1024px  /* Large devices */
xl: 1280px  /* Extra large devices */
```

### Grid System
- **Mobile**: Single column layout
- **Tablet**: 2-column grid for main content
- **Desktop**: 3-column layout with sidebar

## 🔒 Security & Privacy

### Data Protection
- No personal data collection
- API keys stored securely
- HTTPS-only communication
- No data persistence on client

### API Security
- Rate limiting implementation
- Error handling and fallbacks
- Secure API key management
- CORS protection

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Deploy to Netlify
```bash
# Build and deploy
npm run build
netlify deploy --prod --dir=dist
```

### Environment Variables
```bash
# .env.local
VITE_CRICAPI_KEY=your_cricapi_key
VITE_RAPIDAPI_KEY=your_rapidapi_key
```

## 🧪 Testing

### Run Tests
```bash
npm run test
```

### Test Coverage
- Component rendering tests
- Hook functionality tests
- API integration tests
- Accessibility tests

## 📈 Performance Metrics

### Core Web Vitals
- **LCP**: < 2.5s (Largest Contentful Paint)
- **FID**: < 100ms (First Input Delay)
- **CLS**: < 0.1 (Cumulative Layout Shift)

### Bundle Analysis
```bash
npm run build:analyze
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

### Code Standards
- TypeScript for type safety
- ESLint for code quality
- Prettier for formatting
- Conventional commits

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IPL Teams** for inspiration and branding
- **Cricket APIs** for real-time data
- **React Community** for excellent tooling
- **Tailwind CSS** for the design system

## 📞 Support

For questions, issues, or feature requests:

- 📧 Email: support@iplpredictor.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)

## 🔮 Roadmap

### Upcoming Features
- [ ] Player-specific analytics
- [ ] Historical match replay
- [ ] Advanced betting odds
- [ ] Mobile app version
- [ ] Real-time commentary integration
- [ ] Social sharing features

### Version History
- **v1.0.0** - Initial release with basic predictions
- **v1.1.0** - Live data integration
- **v1.2.0** - Enhanced UI and animations
- **v1.3.0** - Advanced ML algorithms

---

**Built with ❤️ for cricket fans worldwide**

*Experience the future of cricket analytics with real-time predictions and beautiful visualizations.*