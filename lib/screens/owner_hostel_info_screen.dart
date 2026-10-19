import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_hostel_map_screen.dart';

class OwnerHostelInfoScreen extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;

  const OwnerHostelInfoScreen({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
  });

  @override
  State<OwnerHostelInfoScreen> createState() => _OwnerHostelInfoScreenState();
}

class _OwnerHostelInfoScreenState extends State<OwnerHostelInfoScreen>
    with SingleTickerProviderStateMixin {
  final _nameCtrl = TextEditingController();
  final _cityCtrl = TextEditingController();
  final _addressCtrl = TextEditingController();

  final _nameFocus = FocusNode();
  final _addressFocus = FocusNode();

  String? _selectedType;
  String? _selectedCity;
  String? _selectedArea;
  String? _error;

  static const List<String> _pakistanCities = [
    'Abbottabad',
    'Attock',
    'Bahawalpur',
    'Bahawalnagar',
    'Bannu',
    'Bhakkar',
    'Burewala',
    'Chakwal',
    'Charsadda',
    'Chiniot',
    'Dera Ghazi Khan',
    'Dera Ismail Khan',
    'Faisalabad',
    'Ghotki',
    'Gilgit',
    'Gujranwala',
    'Gujrat',
    'Gwadar',
    'Haripur',
    'Hafizabad',
    'Hub',
    'Hyderabad',
    'Islamabad',
    'Jacobabad',
    'Jhang',
    'Jhelum',
    'Karachi',
    'Kasur',
    'Khairpur',
    'Khanewal',
    'Khuzdar',
    'Kohat',
    'Kot Addu',
    'Lahore',
    'Larkana',
    'Mandi Bahauddin',
    'Mansehra',
    'Mardan',
    'Mianwali',
    'Mingora',
    'Mirpur',
    'Mirpur Khas',
    'Multan',
    'Muzaffarabad',
    'Murree',
    'Nawabshah',
    'Nowshera',
    'Okara',
    'Pakpattan',
    'Peshawar',
    'Quetta',
    'Rahim Yar Khan',
    'Rawalpindi',
    'Sahiwal',
    'Sargodha',
    'Sheikhupura',
    'Sialkot',
    'Skardu',
    'Sukkur',
    'Swabi',
    'Taxila',
    'Toba Tek Singh',
    'Turbat',
    'Wah Cantt',
    'Wazirabad',
    'Zhob',
  ];

  static const Map<String, List<String>> _cityAreas = {
    'Islamabad': [
      'E-7',
      'E-11',
      'F-6',
      'F-7',
      'F-8',
      'F-10',
      'F-11',
      'G-5',
      'G-6',
      'G-7',
      'G-8',
      'G-9',
      'G-10',
      'G-11',
      'G-13',
      'H-8',
      'H-9',
      'H-10',
      'H-11',
      'H-12',
      'H-13',
      'I-8',
      'I-9',
      'I-10',
      'I-11',
      'I-12',
      'I-14',
      'DHA Phase II',
      'Bahria Town',
      'Other / type exact area',
    ],
    'Rawalpindi': [
      'Saddar',
      'Satellite Town',
      'Chaklala Scheme',
      'Adiala Road',
      'Peshawar Road',
      'Committee Chowk',
      'DHA',
      'Bahria Town',
      'Other / type exact area',
    ],
    'Lahore': [
      'DHA',
      'Gulberg',
      'Johar Town',
      'Model Town',
      'Wapda Town',
      'Bahria Town',
      'Garden Town',
      'Iqbal Town',
      'Raiwind Road',
      'Shadman',
      'Other / type exact area',
    ],
    'Karachi': [
      'Clifton',
      'DHA',
      'Gulshan-e-Iqbal',
      'Gulistan-e-Jauhar',
      'North Nazimabad',
      'PECHS',
      'Saddar',
      'Korangi',
      'Malir',
      'Scheme 33',
      'Other / type exact area',
    ],
    'Faisalabad': [
      'Madina Town',
      'Peoples Colony',
      'Gulberg Colony',
      'D Ground',
      'Jaranwala Road',
      'Sargodha Road',
      'Satiana Road',
      'Other / type exact area',
    ],
    'Multan': [
      'Cantt',
      'Gulgasht Colony',
      'Bosan Road',
      'Mumtazabad',
      'Shah Rukn-e-Alam',
      'Model Town',
      'Other / type exact area',
    ],
    'Peshawar': [
      'University Town',
      'Hayatabad',
      'Saddar',
      'Phase 3',
      'Phase 4',
      'Warsak Road',
      'DHA',
      'Other / type exact area',
    ],
    'Quetta': [
      'Jinnah Town',
      'Cantt',
      'Satellite Town',
      'Brewery Road',
      'Samungli Road',
      'Other / type exact area',
    ],
    'Gujranwala': [
      'Model Town',
      'Satellite Town',
      'Civil Lines',
      'Peoples Colony',
      'Sialkot Road',
      'Other / type exact area',
    ],
    'Sialkot': [
      'Cantt',
      'Model Town',
      'Allama Iqbal Road',
      'Daska Road',
      'Pasrur Road',
      'Other / type exact area',
    ],
    'Hyderabad': [
      'Latifabad',
      'Qasimabad',
      'Cantt',
      'Auto Bhan Road',
      'Hirabad',
      'Other / type exact area',
    ],
  };

  List<String> get _areasForSelectedCity =>
      _cityAreas[_selectedCity] ?? const ['Other / type exact area'];

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  final List<Map<String, dynamic>> _types = [
    {
      'label': 'Girls Hostel',
      'subtitle': 'For female students only',
      'icon': Icons.female_rounded,
      'value': 'girls',
    },
    {
      'label': 'Boys Hostel',
      'subtitle': 'For male students only',
      'icon': Icons.male_rounded,
      'value': 'boys',
    },
  ];

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 750),
    );

    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);

    _slide = Tween<double>(
      begin: 28,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();

    _nameCtrl.dispose();
    _cityCtrl.dispose();
    _addressCtrl.dispose();

    _nameFocus.dispose();
    _addressFocus.dispose();

    super.dispose();
  }

  // ===========================================================================
  // NEXT
  // ===========================================================================

  void _next() {
    FocusScope.of(context).unfocus();

    final name = _nameCtrl.text.trim();
    final city = _cityCtrl.text.trim();
    final address = _addressCtrl.text.trim();

    if (name.isEmpty ||
        city.isEmpty ||
        address.isEmpty ||
        _selectedType == null) {
      setState(() {
        _error = 'Please fill all fields and select a hostel type.';
      });

      return;
    }

    setState(() {
      _error = null;
    });

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerHostelMapScreen(
          ownerEmail: widget.ownerEmail,
          ownerName: widget.ownerName,
          hostelName: name,
          hostelType: _selectedType!,
          city: city,
          area: _selectedArea ?? '',
          address: address,
        ),
      ),
    );
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    final keyboardHeight = MediaQuery.viewInsetsOf(context).bottom;

    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.24,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: LayoutBuilder(
              builder: (context, constraints) {
                return SingleChildScrollView(
                  keyboardDismissBehavior:
                      ScrollViewKeyboardDismissBehavior.onDrag,
                  physics: const BouncingScrollPhysics(),
                  padding: EdgeInsets.fromLTRB(18, 12, 18, 22 + keyboardHeight),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight: constraints.maxHeight - 34,
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildHeader(),

                        const SizedBox(height: 18),

                        AnimatedBuilder(
                          animation: _anim,
                          builder: (context, child) {
                            return Transform.translate(
                              offset: Offset(0, _slide.value),
                              child: Opacity(
                                opacity: _fade.value,
                                child: child,
                              ),
                            );
                          },
                          child: _buildHostelCard(),
                        ),

                        const SizedBox(height: 20),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // HEADER
  // ===========================================================================

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 3),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildBackButton(),

          const SizedBox(height: 16),

          const _StepIndicator(step: 3, total: 9),

          const SizedBox(height: 16),

          Text(
            'Hostel Details',
            style: GoogleFonts.playfairDisplay(
              fontSize: 36,
              height: 1.05,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),

          const SizedBox(height: 8),

          Text(
            'Tell us about the hostel you want to manage.',
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.70),
              fontSize: 13,
              height: 1.4,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBackButton() {
    return InkWell(
      onTap: () => Navigator.maybePop(context),
      borderRadius: BorderRadius.circular(50),
      child: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: const Color(0xFF02383B).withOpacity(0.48),
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white.withOpacity(0.25)),
        ),
        child: const Icon(
          Icons.arrow_back_rounded,
          color: Colors.white,
          size: 20,
        ),
      ),
    );
  }

  // ===========================================================================
  // MAIN DARK TEAL CARD
  // ===========================================================================

  Widget _buildHostelCard() {
    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(maxWidth: 430),
      padding: const EdgeInsets.fromLTRB(20, 17, 20, 20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF063F42).withOpacity(0.88),
            const Color(0xFF023336).withOpacity(0.84),
            const Color(0xFF021F22).withOpacity(0.88),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: AppTheme.tealMint.withOpacity(0.24),
          width: 0.9,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.30),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildCardHeading(),

          const SizedBox(height: 20),

          _buildLabel('Hostel Name'),

          const SizedBox(height: 7),

          _field(
            ctrl: _nameCtrl,
            focusNode: _nameFocus,
            nextFocus: _addressFocus,
            hint: 'Enter hostel name',
            icon: Icons.apartment_outlined,
            capitalization: TextCapitalization.words,
            textInputAction: TextInputAction.next,
          ),

          const SizedBox(height: 17),

          _buildLabel('Hostel Type'),

          const SizedBox(height: 9),

          _buildHostelTypes(),

          const SizedBox(height: 17),

          _buildLabel('City'),

          const SizedBox(height: 7),

          _selectionField(
            value: _selectedCity,
            hint: 'Choose city',
            icon: Icons.location_city_outlined,
            items: _pakistanCities,
            onChanged: (city) {
              setState(() {
                _selectedCity = city;
                _selectedArea = null;
                _cityCtrl.text = city ?? '';
                _error = null;
              });
            },
          ),

          const SizedBox(height: 14),

          _buildLabel('Area / Sector'),

          const SizedBox(height: 7),

          _selectionField(
            value: _selectedArea,
            hint: _selectedCity == null
                ? 'Choose city first'
                : 'Choose area or sector',
            icon: Icons.map_outlined,
            items: _areasForSelectedCity,
            enabled: _selectedCity != null,
            onChanged: (area) {
              setState(() {
                _selectedArea = area;
                _error = null;
              });

              if (area == null || area.startsWith('Other')) {
                return;
              }

              final address = _addressCtrl.text.trim();
              if (address.isEmpty) {
                _addressCtrl.text = area;
              } else if (!address.toLowerCase().contains(area.toLowerCase())) {
                _addressCtrl.text = '$area, $address';
              }
            },
          ),

          const SizedBox(height: 14),

          _buildLabel('Full Address'),

          const SizedBox(height: 7),

          _field(
            ctrl: _addressCtrl,
            focusNode: _addressFocus,
            hint: 'e.g. H-11/4, near PAEC General Hospital',
            icon: Icons.location_on_outlined,
            maxLines: 2,
            capitalization: TextCapitalization.words,
            textInputAction: TextInputAction.done,
            onSubmitted: (_) {
              _next();
            },
          ),

          if (_error != null) ...[const SizedBox(height: 14), _buildError()],

          const SizedBox(height: 16),

          _buildMapInformation(),

          const SizedBox(height: 18),

          _buildNextButton(),
        ],
      ),
    );
  }

  // ===========================================================================
  // CARD HEADING
  // ===========================================================================

  Widget _buildCardHeading() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Container(
          width: 46,
          height: 46,
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.24),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
          ),
          child: const Icon(
            Icons.apartment_rounded,
            size: 22,
            color: AppTheme.tealMint,
          ),
        ),

        const SizedBox(width: 12),

        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Hostel Information',
                style: GoogleFonts.playfairDisplay(
                  color: Colors.white,
                  fontSize: 22,
                  height: 1.05,
                  fontWeight: FontWeight.w700,
                ),
              ),

              const SizedBox(height: 4),

              Text(
                'Step 3 of 9 · Basic hostel details',
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.52),
                  fontSize: 9.7,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // LABEL
  // ===========================================================================

  Widget _buildLabel(String text) {
    return Text(
      text,
      style: GoogleFonts.dmSans(
        color: Colors.white.withOpacity(0.90),
        fontSize: 11.5,
        fontWeight: FontWeight.w700,
      ),
    );
  }

  // ===========================================================================
  // HOSTEL TYPE
  // ===========================================================================

  Widget _buildHostelTypes() {
    return LayoutBuilder(
      builder: (context, constraints) {
        // On a very narrow device, stack cards.
        if (constraints.maxWidth < 300) {
          return Column(
            children: [
              _hostelTypeCard(_types[0]),
              const SizedBox(height: 9),
              _hostelTypeCard(_types[1]),
            ],
          );
        }

        return Row(
          children: [
            Expanded(child: _hostelTypeCard(_types[0])),

            const SizedBox(width: 9),

            Expanded(child: _hostelTypeCard(_types[1])),
          ],
        );
      },
    );
  }

  Widget _hostelTypeCard(Map<String, dynamic> item) {
    final selected = _selectedType == item['value'];

    return InkWell(
      onTap: () {
        setState(() {
          _selectedType = item['value'].toString();

          _error = null;
        });
      },
      borderRadius: BorderRadius.circular(17),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
        decoration: BoxDecoration(
          color: selected
              ? AppTheme.teal.withOpacity(0.30)
              : Colors.black.withOpacity(0.24),
          borderRadius: BorderRadius.circular(17),
          border: Border.all(
            color: selected
                ? AppTheme.tealMint.withOpacity(0.64)
                : Colors.white.withOpacity(0.13),
            width: selected ? 1.2 : 0.8,
          ),
          boxShadow: selected
              ? [
                  BoxShadow(
                    color: AppTheme.teal.withOpacity(0.18),
                    blurRadius: 14,
                  ),
                ]
              : null,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: selected
                        ? AppTheme.tealMint.withOpacity(0.16)
                        : Colors.white.withOpacity(0.07),
                    borderRadius: BorderRadius.circular(11),
                  ),
                  child: Icon(
                    item['icon'] as IconData,
                    color: selected
                        ? AppTheme.tealMint
                        : Colors.white.withOpacity(0.68),
                    size: 19,
                  ),
                ),

                const Spacer(),

                AnimatedContainer(
                  duration: const Duration(milliseconds: 180),
                  width: 22,
                  height: 22,
                  decoration: BoxDecoration(
                    color: selected ? AppTheme.teal : Colors.transparent,
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: selected
                          ? AppTheme.tealMint
                          : Colors.white.withOpacity(0.22),
                    ),
                  ),
                  child: selected
                      ? const Icon(
                          Icons.check_rounded,
                          color: Colors.white,
                          size: 14,
                        )
                      : null,
                ),
              ],
            ),

            const SizedBox(height: 10),

            Text(
              item['label'].toString(),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 11,
                fontWeight: FontWeight.w800,
              ),
            ),

            const SizedBox(height: 3),

            Text(
              item['subtitle'].toString(),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white.withOpacity(0.48),
                fontSize: 8.3,
                height: 1.3,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // INPUT FIELD
  // ===========================================================================

  Widget _selectionField({
    required String? value,
    required String hint,
    required IconData icon,
    required List<String> items,
    required ValueChanged<String?> onChanged,
    bool enabled = true,
  }) {
    return Container(
      height: 54,
      decoration: BoxDecoration(
        color: const Color(0xFFF7F9F7),
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: const Color(0xFFD8E0DD), width: 0.8),
      ),
      child: Row(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 15),
            child: Icon(
              icon,
              size: 19,
              color: enabled
                  ? const Color(0xFF176B6D)
                  : const Color(0xFF9AA9A8),
            ),
          ),
          Expanded(
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: value,
                isExpanded: true,
                icon: Icon(
                  Icons.keyboard_arrow_down_rounded,
                  color: enabled
                      ? const Color(0xFF176B6D)
                      : const Color(0xFF9AA9A8),
                ),
                padding: const EdgeInsets.only(right: 12),
                hint: Text(
                  hint,
                  style: GoogleFonts.dmSans(
                    color: const Color(0xFF6C7E7D),
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                selectedItemBuilder: (context) => items
                    .map(
                      (item) => Align(
                        alignment: Alignment.centerLeft,
                        child: Text(
                          item,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.dmSans(
                            color: const Color(0xFF062F32),
                            fontSize: 14.5,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    )
                    .toList(),
                items: items
                    .map(
                      (item) => DropdownMenuItem<String>(
                        value: item,
                        child: Text(
                          item,
                          style: GoogleFonts.dmSans(
                            color: const Color(0xFF062F32),
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    )
                    .toList(),
                onChanged: enabled ? onChanged : null,
                dropdownColor: const Color(0xFFF7F9F7),
                borderRadius: BorderRadius.circular(14),
                menuMaxHeight: 300,
                iconSize: 23,
                alignment: Alignment.centerLeft,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _field({
    required TextEditingController ctrl,
    required FocusNode focusNode,
    required String hint,
    required IconData icon,
    FocusNode? nextFocus,
    TextInputAction? textInputAction,
    TextCapitalization capitalization = TextCapitalization.none,
    int maxLines = 1,
    ValueChanged<String>? onSubmitted,
  }) {
    return AnimatedBuilder(
      animation: focusNode,
      builder: (context, child) {
        final focused = focusNode.hasFocus;

        return AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          decoration: BoxDecoration(
            // White field, just like requested.
            color: const Color(0xFFF7F9F7),
            borderRadius: BorderRadius.circular(15),
            border: Border.all(
              color: focused ? AppTheme.teal : const Color(0xFFD8E0DD),
              width: focused ? 1.2 : 0.8,
            ),
            boxShadow: focused
                ? [
                    BoxShadow(
                      color: AppTheme.teal.withOpacity(0.14),
                      blurRadius: 10,
                    ),
                  ]
                : null,
          ),
          child: TextField(
            controller: ctrl,
            focusNode: focusNode,
            maxLines: maxLines,
            textCapitalization: capitalization,
            textInputAction: textInputAction,
            onSubmitted:
                onSubmitted ??
                (_) {
                  if (nextFocus != null) {
                    FocusScope.of(context).requestFocus(nextFocus);
                  }
                },
            style: GoogleFonts.dmSans(
              fontSize: 14.5,
              fontWeight: FontWeight.w700,

              // Input text DARK.
              color: const Color(0xFF062F32),
            ),
            decoration: InputDecoration(
              hintText: hint,
              hintStyle: GoogleFonts.dmSans(
                color: const Color(0xFF889897),
                fontSize: 12.3,
              ),

              // Icon remains DARK TEAL.
              prefixIcon: Icon(
                icon,
                size: 19,
                color: focused ? AppTheme.teal : const Color(0xFF176B6D),
              ),

              border: InputBorder.none,

              contentPadding: const EdgeInsets.symmetric(
                horizontal: 14,
                vertical: 15,
              ),
            ),
          ),
        );
      },
    );
  }

  // ===========================================================================
  // ERROR
  // ===========================================================================

  Widget _buildError() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF7F2525).withOpacity(0.42),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFFF7777).withOpacity(0.38)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.error_outline_rounded,
            color: Color(0xFFFF8A8A),
            size: 16,
          ),

          const SizedBox(width: 8),

          Expanded(
            child: Text(
              _error!,
              style: GoogleFonts.dmSans(
                color: const Color(0xFFFFB5B5),
                fontSize: 10.3,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // MAP INFORMATION
  // ===========================================================================

  Widget _buildMapInformation() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 11),
      decoration: BoxDecoration(
        color: const Color(0xFF01272A).withOpacity(0.54),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.24),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(
              Icons.location_on_outlined,
              color: AppTheme.tealMint,
              size: 18,
            ),
          ),

          const SizedBox(width: 10),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Hostel location comes next',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 10.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  'Mark the hostel building, gate and entrance on the map in the next step.',
                  style: GoogleFonts.dmSans(
                    color: Colors.white.withOpacity(0.56),
                    fontSize: 9.3,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // NEXT BUTTON
  // ===========================================================================

  Widget _buildNextButton() {
    return SizedBox(
      width: double.infinity,
      height: 50,
      child: ElevatedButton(
        onPressed: _next,
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.teal,
          foregroundColor: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.map_outlined, size: 18),

            const SizedBox(width: 8),

            Flexible(
              child: Text(
                'Next: Pin on Map',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  fontSize: 14.5,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),

            const SizedBox(width: 8),

            Container(
              width: 25,
              height: 25,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.15),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.arrow_forward_rounded, size: 15),
            ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// STEP INDICATOR
// =============================================================================

class _StepIndicator extends StatelessWidget {
  final int step;
  final int total;

  const _StepIndicator({required this.step, required this.total});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: List.generate(total, (index) {
        final completed = index < step;

        final current = index == step - 1;

        return Expanded(
          child: Container(
            height: current ? 4 : 3,
            margin: EdgeInsets.only(right: index < total - 1 ? 4 : 0),
            decoration: BoxDecoration(
              color: completed
                  ? AppTheme.tealMint
                  : Colors.white.withOpacity(0.22),
              borderRadius: BorderRadius.circular(3),
              boxShadow: current
                  ? [
                      BoxShadow(
                        color: AppTheme.tealMint.withOpacity(0.25),
                        blurRadius: 5,
                      ),
                    ]
                  : null,
            ),
          ),
        );
      }),
    );
  }
}
